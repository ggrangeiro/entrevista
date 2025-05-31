import cv2
import mediapipe as mp
import numpy as np

print("Iniciando o Reconhecedor de Gestos (com Contagem de Dedos, Debouncing)...")

# --- Constantes e Configurações ---
OK_SIGN_DISTANCE_THRESHOLD = 0.06
GESTURE_COLORS = {
    "Nenhum gesto": (255, 255, 255), # Branco
    "Mao Aberta": (0, 255, 0),      # Verde
    "Sinal de Paz": (255, 0, 0),    # Azul
    "Punho Fechado": (0, 0, 255),   # Vermelho
    "Apontando (Indicador)": (0, 255, 255), # Amarelo
    "Polegar para Cima": (255, 0, 255), # Magenta
    "OK": (0, 165, 255),             # Laranja
    "Contando: 1": (200, 100, 50),   # Cores para contagem
    "Contando: 2": (200, 150, 50),
    "Contando: 3": (200, 200, 50),
    "Contando: 4": (200, 250, 50),
    # "Contando: 5" é coberto por "Mao Aberta"
    # "Contando: 0" pode ser coberto por "Punho Fechado" ou "Nenhum gesto"
}
DEFAULT_GESTURE_COLOR = (200, 200, 200) 

FRAMES_TO_CONFIRM_GESTURE = 5

# --- Inicialização do MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Captura de Vídeo ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO: Não foi possível abrir a câmera.")
    exit()

print(f"Câmera aberta. Gestos serão confirmados após {FRAMES_TO_CONFIRM_GESTURE} frames. Pressione 'q' para sair.")

confirmed_gesture = "Nenhum gesto"
last_seen_raw_gesture = "Nenhum gesto"
frames_gesture_seen_count = 0
debug_finger_states = ""

def calculate_normalized_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# --- Loop Principal de Processamento ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2] 
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False
    results = hands_detector.process(image_rgb)
    image_rgb.flags.writeable = True

    raw_gesture_this_frame = "Nenhum gesto"
    debug_finger_states = "No hand" 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=frame, landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

            try:
                landmarks = hand_landmarks.landmark
                
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
                thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
                ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
                ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
                pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
                pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

                threshold_extension = 0.01 
                threshold_thumb_up = 0.03  
                
                index_extended = index_tip.y < (index_pip.y - threshold_extension)
                middle_extended = middle_tip.y < (middle_pip.y - threshold_extension)
                ring_extended = ring_tip.y < (ring_pip.y - threshold_extension)
                pinky_extended = pinky_tip.y < (pinky_pip.y - threshold_extension)
                thumb_generally_extended = thumb_tip.y < (thumb_ip.y - threshold_extension)
                thumb_is_up = thumb_tip.y < (thumb_mcp.y - threshold_thumb_up)
                index_bent_fist = index_tip.y > index_mcp.y 
                middle_bent_fist = middle_tip.y > middle_mcp.y
                ring_bent_fist = ring_tip.y > ring_mcp.y
                pinky_bent_fist = pinky_tip.y > pinky_mcp.y
                thumb_bent_fist = thumb_tip.y > thumb_ip.y
                ring_bent_peace = ring_tip.y > ring_pip.y
                pinky_bent_peace = pinky_tip.y > pinky_pip.y
                distance_thumb_index_tip = calculate_normalized_distance(thumb_tip, index_tip)
                ok_sign_proximity = distance_thumb_index_tip < OK_SIGN_DISTANCE_THRESHOLD
                
                debug_finger_states = (
                    f"T_gen_ext:{int(thumb_generally_extended)} T_up:{int(thumb_is_up)} T_bent_F:{int(thumb_bent_fist)} | "
                    f"I_ext:{int(index_extended)} I_bent_F:{int(index_bent_fist)} | "
                    f"M_ext:{int(middle_extended)} M_bent_F:{int(middle_bent_fist)} | "
                    f"R_ext:{int(ring_extended)} R_bent_P:{int(ring_bent_peace)} R_bent_F:{int(ring_bent_fist)} | "
                    f"P_ext:{int(pinky_extended)} P_bent_P:{int(pinky_bent_peace)} P_bent_F:{int(pinky_bent_fist)} "
                    f"OK_prox:{int(ok_sign_proximity)} (d:{distance_thumb_index_tip:.2f})"
                )

                # --- Lógica de Detecção de Gestos (atribui a raw_gesture_this_frame) ---
                if index_extended and middle_extended and ring_extended and pinky_extended and thumb_generally_extended:
                    raw_gesture_this_frame = "Mao Aberta"
                elif ok_sign_proximity and middle_extended and ring_extended and pinky_extended:
                    raw_gesture_this_frame = "OK"
                elif thumb_is_up and \
                     index_bent_fist and middle_bent_fist and \
                     ring_bent_fist and pinky_bent_fist:
                    raw_gesture_this_frame = "Polegar para Cima"
                elif index_extended and \
                     middle_bent_fist and ring_bent_fist and pinky_bent_fist and \
                     thumb_bent_fist: 
                    raw_gesture_this_frame = "Apontando (Indicador)"
                elif index_extended and \
                     middle_extended and \
                     not ring_extended and not pinky_extended and \
                     (thumb_bent_fist or not thumb_generally_extended): # Polegar não pode estar esticado para cima como em "Mão Aberta"
                    raw_gesture_this_frame = "Sinal de Paz" # Ajustado para ser mais específico
                elif index_bent_fist and \
                     middle_bent_fist and \
                     ring_bent_fist and \
                     pinky_bent_fist and \
                     thumb_bent_fist: 
                    raw_gesture_this_frame = "Punho Fechado"
                else:
                    # Se nenhum gesto específico foi detectado, contamos os dedos esticados
                    extended_fingers_count = 0
                    if thumb_generally_extended: extended_fingers_count += 1
                    if index_extended: extended_fingers_count += 1
                    if middle_extended: extended_fingers_count += 1
                    if ring_extended: extended_fingers_count += 1
                    if pinky_extended: extended_fingers_count += 1
                    
                    if extended_fingers_count > 0 and extended_fingers_count < 5 : # Evitar sobrescrever "Mao Aberta"
                        raw_gesture_this_frame = f"Contando: {extended_fingers_count}"
                    # Se 0 dedos esticados e não é "Punho Fechado", ou 5 e não é "Mao Aberta" (improvável com a lógica atual)
                    # ele permanecerá "Nenhum gesto".

            except Exception as e:
                debug_finger_states = "Erro nos landmarks"
                # print(f"Erro ao acessar landmarks para gesto: {e}") 
                pass
    
    # --- Lógica de Debouncing ---
    if raw_gesture_this_frame == last_seen_raw_gesture:
        frames_gesture_seen_count += 1
    else:
        frames_gesture_seen_count = 1 
        last_seen_raw_gesture = raw_gesture_this_frame

    if frames_gesture_seen_count >= FRAMES_TO_CONFIRM_GESTURE:
        confirmed_gesture = last_seen_raw_gesture
    
    text_color = GESTURE_COLORS.get(confirmed_gesture, DEFAULT_GESTURE_COLOR)

    cv2.putText(frame, confirmed_gesture, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    cv2.putText(frame, debug_finger_states, (10, frame_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Reconhecedor de Gestos com MediaPipe', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

# --- Finalização ---
print("Finalizando o script...")
cap.release()
if 'hands_detector' in locals() and hands_detector:
    hands_detector.close()
cv2.destroyAllWindows()
print("Recursos liberados e janelas destruídas.")