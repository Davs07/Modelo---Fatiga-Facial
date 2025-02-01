import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist
import tensorflow as tf
import csv
from datetime import datetime


# ==============================
# Funciones Auxiliares
# ==============================

def eye_aspect_ratio(eye_points):
    """
    Calcula la relación de aspecto del ojo (EAR).

    Args:
        eye_points (list of tuples): Coordenadas (x, y) de los puntos del ojo.

    Returns:
        float: EAR calculado.
    """
    # Calcular distancias verticales
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Calcular distancia horizontal
    C = dist.euclidean(eye_points[0], eye_points[3])
    # Calcular EAR
    ear = (A + B) / (2.0 * C) if C != 0 else 0
    return ear


def mouth_aspect_ratio(mouth_points):
    """
    Calcula la relación de aspecto de la boca (MAR).

    Args:
        mouth_points (list of tuples): Coordenadas (x, y) de los puntos de la boca.

    Returns:
        float: MAR calculado.
    """
    # Calcular distancias verticales
    A = dist.euclidean(mouth_points[13], mouth_points[19])  # Distancia entre punto superior e inferior
    B = dist.euclidean(mouth_points[14], mouth_points[18])
    C = dist.euclidean(mouth_points[15], mouth_points[17])
    # Calcular distancia horizontal
    D = dist.euclidean(mouth_points[0], mouth_points[6])
    # Calcular MAR
    mar = (A + B + C) / (3.0 * D) if D != 0 else 0
    return mar


def preprocess_face_roi(frame, landmarks):
    """
    Extrae y preprocesa la región facial usando Face Mesh.

    Args:
        frame (ndarray): Imagen del frame capturado.
        landmarks (LandmarkList): Puntos de referencia faciales detectados.

    Returns:
        tuple: ROI preprocesado y coordenadas del rectángulo (x_min, y_min, x_max, y_max).
    """
    # Obtener coordenadas del rostro
    x_coords = [lm.x * frame.shape[1] for lm in landmarks.landmark]
    y_coords = [lm.y * frame.shape[0] for lm in landmarks.landmark]

    # Expandir área del rostro
    expand_factor = 1.3
    x_min = max(0, int(min(x_coords) * (2 - expand_factor)))
    y_min = max(0, int(min(y_coords) * (2 - expand_factor)))
    x_max = min(frame.shape[1], int(max(x_coords) * expand_factor))
    y_max = min(frame.shape[0], int(max(y_coords) * expand_factor))

    # Extraer ROI y preprocesar
    face_roi = frame[y_min:y_max, x_min:x_max]
    if face_roi.size == 0:
        return None, x_min, y_min, x_max, y_max

    processed = cv2.resize(face_roi, (224, 224))
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return processed / 255.0, x_min, y_min, x_max, y_max


def calculate_fatigue_score(model_pred, ear, mar, blink_duration):
    """
    Combina métricas para calcular la puntuación de fatiga.

    Args:
        model_pred (float): Predicción del modelo.
        ear (float): Relación de aspecto del ojo.
        mar (float): Relación de aspecto de la boca.
        blink_duration (float): Duración del parpadeo.

    Returns:
        float: Puntuación de fatiga (0.0 a 1.0).
    """
    score = 0.0

    # Contribución del modelo
    score += (0.4 * (1 - model_pred)) if model_pred < MODEL_THRESH else 0

    # Contribución de ojos
    if ear < EYE_AR_THRESH:
        score += 0.2
    if blink_duration > 0.5:
        score += 0.15 * min(blink_duration, 2.0)

    # Contribución de boca
    if mar > MOUTH_AR_THRESH:
        score += 0.25

    return min(score, 1.0)


# ==============================
# Configuración Inicial
# ==============================

# Ruta al modelo preentrenado
MODEL_PATH = r'C:\Users\davyr\Downloads\my_model.h5'

# Cargar modelo preentrenado
model = tf.keras.models.load_model(MODEL_PATH)

# Configuración MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Constantes ajustadas
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 4
MOUTH_AR_THRESH = 0.75
MODEL_THRESH = 0.5  # Umbral para el modelo
FATIGUE_CONSEC_FRAMES = 25  # Frames consecutivos para alerta final

# Variables de estado
COUNTER_EYE = 0
COUNTER_MOUTH = 0
FATIGUE_COUNTER = 0
cooldown = 0
ear_history = []
blink_start_time = None
current_blink_duration = 0.0

# Variables acumulativas para contar parpadeos y bostezos
total_blinks = 0
total_yawns = 0

# Configurar el archivo CSV para almacenar el historial de datos
CSV_FILE = 'historial_fatiga.csv'

# Crear y escribir encabezados si el archivo no existe
try:
    with open(CSV_FILE, mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Timestamp', 'Parpadeos', 'Bostezos', 'EAR Promedio', 'MAR Promedio', 'Puntuación Fatiga Promedio'])
except FileExistsError:
    # El archivo ya existe
    pass

# Variables para almacenar temporalmente los datos del historial
historial = {
    'timestamp': [],
    'parpadeos': [],
    'bostezos': [],
    'ear_promedio': [],
    'mar_promedio': [],
    'fatigue_promedio': []
}

# Intervalo de tiempo para registrar en el historial (en segundos)
INTERVALO_HISTORIAL = 5  # Puedes ajustar este valor según tus necesidades
ultimo_registro = time.time()

# ==============================
# Inicialización de la Captura de Video
# ==============================

cap = cv2.VideoCapture(0)  # Cambia el índice si usas una cámara diferente

# ==============================
# Bucle Principal
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Voltear la imagen horizontalmente
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    fatigue_data = {
        'model_pred': 0.0,
        'ear': 0.0,
        'mar': 0.0,
        'blink_duration': 0.0
    }

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Preprocesar ROI facial para el modelo
            face_roi, x_min, y_min, x_max, y_max = preprocess_face_roi(frame, face_landmarks)
            if face_roi is not None:
                fatigue_data['model_pred'] = model.predict(np.expand_dims(face_roi, axis=0), verbose=0)[0][0]

            # Detección de ojos izquierdos y derechos
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            left_eye = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                        for lm in [face_landmarks.landmark[i] for i in left_eye_indices]]
            right_eye = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                         for lm in [face_landmarks.landmark[i] for i in right_eye_indices]]

            # Calcular EAR para ambos ojos
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            fatigue_data['ear'] = (leftEAR + rightEAR) / 2.0

            # Lógica de parpadeos
            ear_history.append(fatigue_data['ear'])
            if len(ear_history) > 5:
                ear_history.pop(0)
            smoothed_ear = np.mean(ear_history)

            if cooldown > 0:
                cooldown -= 1
            else:
                if smoothed_ear < EYE_AR_THRESH:
                    if COUNTER_EYE == 0:
                        blink_start_time = time.time()
                    COUNTER_EYE += 1
                else:
                    if COUNTER_EYE >= EYE_AR_CONSEC_FRAMES:
                        current_blink_duration = time.time() - blink_start_time
                        cooldown = EYE_AR_CONSEC_FRAMES
                        total_blinks += 1  # Incrementar el contador total de parpadeos
                    COUNTER_EYE = 0
                    blink_start_time = None

            fatigue_data['blink_duration'] = current_blink_duration if blink_start_time else 0.0

            # Detección de bostezos
            mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405,
                             321, 375, 78, 191, 80, 81, 82, 13,
                             19, 14, 18, 15, 17, 12, 16]
            mouth = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                     for lm in [face_landmarks.landmark[i] for i in mouth_indices]]
            if len(mouth) >= 20:
                fatigue_data['mar'] = mouth_aspect_ratio(mouth)

            # Lógica de bostezos
            if fatigue_data['mar'] > MOUTH_AR_THRESH:
                COUNTER_MOUTH += 1
                if COUNTER_MOUTH == 1:
                    total_yawns += 1  # Incrementar el contador total de bostezos
            else:
                COUNTER_MOUTH = 0

            # Cálculo de fatiga integrada
            fatigue_score = calculate_fatigue_score(
                fatigue_data['model_pred'],
                fatigue_data['ear'],
                fatigue_data['mar'],
                fatigue_data['blink_duration']
            )

            # Lógica de alerta
            if fatigue_score > 0.65:
                FATIGUE_COUNTER += 1
                alert_color = (0, 0, 255)  # Rojo
                alert_text = "ALERTA DE FATIGA!"
            else:
                FATIGUE_COUNTER = max(0, FATIGUE_COUNTER - 2)
                alert_color = (0, 255, 0)  # Verde
                alert_text = "Estado normal"

            # Dibujar rectángulo alrededor de la cara
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Mostrar información de fatiga en la parte superior izquierda
            cv2.putText(frame, f"Fatiga: {fatigue_score:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)
            cv2.putText(frame, alert_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

            if FATIGUE_COUNTER >= FATIGUE_CONSEC_FRAMES:
                cv2.putText(frame, "ACCION REQUERIDA!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Aquí podrías agregar una acción adicional (sonido, notificación, etc.)

            # ==============================
            # Mostrar Métricas en el Lado Derecho
            # ==============================
            metrics_x = frame.shape[1] - 250  # Posición X para las métricas
            metrics_y = 30  # Posición Y inicial

            # Parpadeos en Intervalo
            cv2.putText(frame, f"Parpadeos Intervalo: {COUNTER_EYE}", (metrics_x, metrics_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            metrics_y += 30

            # Parpadeos Totales
            cv2.putText(frame, f"Parpadeos Totales: {total_blinks}", (metrics_x, metrics_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            metrics_y += 30

            # Ojos Cerrados (EAR)
            cv2.putText(frame, f"Ojos Cerrados: {smoothed_ear:.2f}", (metrics_x, metrics_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            metrics_y += 30

            # Bostezo en Intervalo
            btozo_text = "Sí" if fatigue_data['mar'] > MOUTH_AR_THRESH else "No"
            cv2.putText(frame, f"Bostezo Intervalo: {COUNTER_MOUTH}", (metrics_x, metrics_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            metrics_y += 30

            # Bostezo Total
            cv2.putText(frame, f"Bostezo Totales: {total_yawns}", (metrics_x, metrics_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            metrics_y += 30

            # Puntuación de Fatiga
            cv2.putText(frame, f"Puntuación Fatiga: {fatigue_score:.2f}", (metrics_x, metrics_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # ==============================
            # Imprimir Métricas en la Consola
            # ==============================
            print(f"Parpadeos: {COUNTER_EYE}")
            print(f"Ojos Cerrados (EAR): {smoothed_ear:.2f}")
            print(f"Bostezo: {'Sí' if fatigue_data['mar'] > MOUTH_AR_THRESH else 'No'}")
            print(f"Puntuación de Fatiga: {fatigue_score:.2f}")
            print(f"Parpadeos Totales: {total_blinks}")
            print(f"Bostezo Totales: {total_yawns}")
            print("-----")

            # ==============================
            # Registrar Datos en el Historial
            # ==============================
            tiempo_actual = time.time()
            if tiempo_actual - ultimo_registro >= INTERVALO_HISTORIAL:
                timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                historial['timestamp'].append(timestamp)
                historial['parpadeos'].append(COUNTER_EYE)
                historial['bostezos'].append(1 if fatigue_data['mar'] > MOUTH_AR_THRESH else 0)
                historial['ear_promedio'].append(smoothed_ear)
                historial['mar_promedio'].append(fatigue_data['mar'])
                historial['fatigue_promedio'].append(fatigue_score)

                # Escribir en el archivo CSV
                with open(CSV_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        timestamp,
                        COUNTER_EYE,
                        1 if fatigue_data['mar'] > MOUTH_AR_THRESH else 0,
                        f"{smoothed_ear:.2f}",
                        f"{fatigue_data['mar']:.2f}",
                        f"{fatigue_score:.2f}"
                    ])

                # Reiniciar contadores si es necesario
                # COUNTER_EYE = 0
                # Puedes descomentar la línea anterior si deseas reiniciar el contador de parpadeos después de cada registro

                ultimo_registro = tiempo_actual

    # Mostrar el video con las métricas superpuestas
    cv2.imshow('Sistema Integrado de Detección de Fatiga', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# Liberación de Recursos
# ==============================

cap.release()
cv2.destroyAllWindows()