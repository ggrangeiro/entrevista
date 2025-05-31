import cv2
import mediapipe as mp
import numpy as np

print("Iniciando o Alerta de Postura (com Verificação de Ombros)...")

# --- Constantes e Configurações ---
# Para Cabeça Caída
HEAD_DROOP_THRESHOLD_Y_OFFSET = 0.08 # Quão abaixo dos ombros o nariz precisa estar

# Para Ombros Desnivelados
SHOULDER_Y_DIFF_THRESHOLD = 0.03 # Diferença Y normalizada máx. entre ombros (ex: 3% da altura do frame)

# Para Ombros Encolhidos (distância vertical mínima entre orelha e ombro)
SHOULDER_HUNCH_THRESHOLD_MIN_DIST = 0.05 # Ombros devem estar pelo menos X% abaixo das orelhas

# Frames consecutivos em má postura para disparar o alerta
BAD_POSTURE_CONSEC_FRAMES = 60 

# --- Inicialização do MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False, model_complexity=1, smooth_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Captura de Vídeo ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO: Não foi possível abrir a câmera.")
    exit()

print(f"Câmera aberta. Monitore a postura. Alerta após {BAD_POSTURE_CONSEC_FRAMES} frames. Pressione 'q' para sair.")

bad_posture_counter = 0
show_posture_alert = False
debug_posture_states_text = "Aguardando landmarks..." # Para depuração na tela

# --- Loop Principal de Processamento ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("ERRO: Falha ao capturar frame ou fim do stream de vídeo.")
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2] 
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False
    results = pose_detector.process(image_rgb)
    image_rgb.flags.writeable = True

    is_bad_posture_this_frame = False
    debug_posture_states_text = "Landmarks não totalmente visíveis"


    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        try:
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            # left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] # Para futuras melhorias
            # right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] # Para futuras melhorias

            # Verificar visibilidade dos landmarks chave
            if nose.visibility > 0.5 and \
               left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and \
               left_ear.visibility > 0.5 and right_ear.visibility > 0.5:

                shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
                ear_mid_y = (left_ear.y + right_ear.y) / 2.0
                
                # 1. Condição de Cabeça Caída
                is_head_drooped = nose.y > (shoulder_mid_y + HEAD_DROOP_THRESHOLD_Y_OFFSET)
                
                # 2. Condição de Ombros Desnivelados
                shoulder_y_difference = abs(left_shoulder.y - right_shoulder.y)
                are_shoulders_uneven = shoulder_y_difference > SHOULDER_Y_DIFF_THRESHOLD

                # 3. Condição de Ombros Encolhidos (muito próximos das orelhas verticalmente)
                # Distância vertical entre a média das orelhas e a média dos ombros.
                # Se os ombros estiverem abaixo das orelhas, esta distância é positiva.
                # Se for muito pequena, estão encolhidos.
                vertical_dist_ear_to_shoulder = shoulder_mid_y - ear_mid_y
                are_shoulders_hunched = vertical_dist_ear_to_shoulder < SHOULDER_HUNCH_THRESHOLD_MIN_DIST
                                
                # Combinar condições para má postura
                is_bad_posture_this_frame = is_head_drooped or are_shoulders_uneven or are_shoulders_hunched

                # --- Depuração Visual e Textual ---
                # Linhas para Cabeça Caída
                cv2.circle(frame, (int(nose.x * frame_width), int(nose.y * frame_height)), 5, (0, 255, 255), -1) # Nariz
                cv2.line(frame, (0, int(shoulder_mid_y * frame_height)), (frame_width, int(shoulder_mid_y * frame_height)), (255, 0, 0), 1) # Linha ombros
                cv2.line(frame, (0, int((shoulder_mid_y + HEAD_DROOP_THRESHOLD_Y_OFFSET) * frame_height)), 
                         (frame_width, int((shoulder_mid_y + HEAD_DROOP_THRESHOLD_Y_OFFSET) * frame_height)), 
                         (0, 0, 255), 1) # Linha limiar cabeça caída

                # Linha entre ombros para visualizar desnivelamento
                uneven_color = (0, 165, 255) if are_shoulders_uneven else (0, 255, 0) # Laranja se desnivelado, verde se ok
                cv2.line(frame, (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height)),
                         (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height)), uneven_color, 2)

                debug_posture_states_text = (f"Drop:{int(is_head_drooped)} "
                                             f"Uneven:{int(are_shoulders_uneven)}({shoulder_y_difference:.2f}) "
                                             f"Hunch:{int(are_shoulders_hunched)}({vertical_dist_ear_to_shoulder:.2f})")
            else:
                debug_posture_states_text = "Alguns landmarks chave nao visiveis"

        except Exception as e:
            debug_posture_states_text = "Erro ao processar landmarks"
            # print(f"Erro ao acessar landmarks: {e}")
            pass

    # --- Lógica de Alerta Sustentado ---
    if is_bad_posture_this_frame:
        bad_posture_counter += 1
    else:
        bad_posture_counter = 0 
        show_posture_alert = False 

    if bad_posture_counter >= BAD_POSTURE_CONSEC_FRAMES:
        show_posture_alert = True
    
    if show_posture_alert:
        alert_text_size = cv2.getTextSize("CORRIJA A POSTURA!", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame_width - alert_text_size[0]) // 2
        text_y = (frame_height + alert_text_size[1]) // 2
        cv2.putText(frame, "CORRIJA A POSTURA!", (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Exibir texto de depuração
    cv2.putText(frame, debug_posture_states_text, (10, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Alerta de Postura com MediaPipe Pose', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

# --- Finalização ---
print("Finalizando o script...")
cap.release()
if 'pose_detector' in locals() and pose_detector:
    pose_detector.close()
cv2.destroyAllWindows()
print("Recursos liberados e janelas destruídas.")