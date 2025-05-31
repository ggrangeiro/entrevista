import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist # Para calcular a distância euclidiana

print("Iniciando o Contador de Piscadas com MediaPipe...")

# --- Constantes e Configurações ---

# Limiar do Eye Aspect Ratio (EAR) para indicar uma piscada
EYE_AR_THRESH = 0.15  # Ajuste este valor conforme necessário (entre 0.2 e 0.3 é comum)
# Número de frames consecutivos que o olho deve estar abaixo do limiar para contar como piscada
EYE_AR_CONSEC_FRAMES = 2 # Ajuste este valor conforme necessário

# Inicializar o contador de piscadas e o contador de frames para o EAR
blink_counter = 0
ear_frame_counter = 0

# Índices dos marcos faciais do MediaPipe para os olhos
# Estes são os 6 pontos para cada olho usados no cálculo do EAR (P1 a P6)
# A ordem é importante para a fórmula do EAR: P1(canto externo), P2, P3 (pálpebra superior), P4(canto interno), P5, P6 (pálpebra inferior)
# Estes índices são baseados no mapa de marcos do MediaPipe Face Mesh. VERIFIQUE se são os ideais para você.

# Olho ESQUERDO da pessoa (aparece à DIREITA na imagem da câmera se não espelhada)
LEFT_EYE_POINTS_FOR_EAR = [362, 385, 387, 263, 373, 380]
# Olho DIREITO da pessoa (aparece à ESQUERDA na imagem da câmera se não espelhada)
RIGHT_EYE_POINTS_FOR_EAR = [33, 160, 158, 133, 153, 144]


# --- Funções Auxiliares ---

def calculate_ear(eye_points_pixel):
    """
    Calcula o Eye Aspect Ratio (EAR) para um olho.
    eye_points_pixel: Uma lista ou array NumPy de 6 tuplas (x,y) representando
                      os marcos P1 a P6 do olho em coordenadas de pixel.
    """
    if len(eye_points_pixel) != 6:
        return 0.0 # Ou levante um erro

    # Mapeie os pontos para p1-p6 para clareza com a fórmula do EAR
    p1, p2, p3, p4, p5, p6 = eye_points_pixel

    # Distâncias verticais (entre pálpebras)
    A = dist.euclidean(p2, p6) # Distância entre P2 e P6
    B = dist.euclidean(p3, p5) # Distância entre P3 e P5

    # Distância horizontal (entre cantos do olho)
    C = dist.euclidean(p1, p4) # Distância entre P1 e P4

    if C == 0: # Evitar divisão por zero se os pontos do canto coincidirem
        return 0.0 

    # Cálculo do EAR
    ear_val = (A + B) / (2.0 * C)
    return ear_val

# --- Inicialização do MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,                # Detectar apenas um rosto
    static_image_mode=False,        # Ideal para vídeo
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# --- Captura de Vídeo ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO: Não foi possível abrir a câmera.")
    exit()

print("Câmera aberta. Pressione 'q' para sair.")

# --- Loop Principal de Processamento ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("ERRO: Falha ao capturar frame ou fim do stream de vídeo.")
        break

    # Opcional: Espelhar o frame para uma visualização mais natural (como um espelho)
    frame = cv2.flip(frame, 1)

    # Obter dimensões do frame para desnormalizar os marcos
    frame_height, frame_width = frame.shape[:2]

    # Converter BGR para RGB para o MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Para melhorar o desempenho, opcionalmente marque a imagem como não gravável
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True # Marque como gravável novamente

    ear_avg = 0.0 # Inicializar EAR médio

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: # Deve haver apenas um rosto devido a max_num_faces=1
            
            # Extrair coordenadas dos olhos e calcular EAR
            # Olho ESQUERDO (da pessoa)
            left_eye_coords_pixel = []
            for idx in LEFT_EYE_POINTS_FOR_EAR:
                lm = face_landmarks.landmark[idx]
                coord_x = int(lm.x * frame_width)
                coord_y = int(lm.y * frame_height)
                left_eye_coords_pixel.append((coord_x, coord_y))
            
            if len(left_eye_coords_pixel) == 6:
                ear_left = calculate_ear(np.array(left_eye_coords_pixel))
            else:
                ear_left = 0.0

            # Olho DIREITO (da pessoa)
            right_eye_coords_pixel = []
            for idx in RIGHT_EYE_POINTS_FOR_EAR:
                lm = face_landmarks.landmark[idx]
                coord_x = int(lm.x * frame_width)
                coord_y = int(lm.y * frame_height)
                right_eye_coords_pixel.append((coord_x, coord_y))

            if len(right_eye_coords_pixel) == 6:
                ear_right = calculate_ear(np.array(right_eye_coords_pixel))
            else:
                ear_right = 0.0
            
            # EAR médio dos dois olhos
            ear_avg = (ear_left + ear_right) / 2.0

            # Lógica de Detecção de Piscada
            if ear_avg < EYE_AR_THRESH:
                ear_frame_counter += 1
            else:
                if ear_frame_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                ear_frame_counter = 0 # Resetar contador de frames do olho fechado

            # Desenhar os marcos faciais (opcional, mas bom para depuração)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    
    # Exibir o EAR e o contador de piscadas no frame
    cv2.putText(frame, f"EAR: {ear_avg:.2f}", (frame_width - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Piscadas: {blink_counter}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar o frame resultante
    cv2.imshow('Contador de Piscadas - MediaPipe', frame)

    # Sair com a tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Finalização ---
print(f"Saiu do loop principal. Total de piscadas detectadas: {blink_counter}")
cap.release()
if 'face_mesh' in locals() and face_mesh:
    face_mesh.close()
cv2.destroyAllWindows()
print("Script finalizado.")