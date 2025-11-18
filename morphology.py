import cv2
import numpy as np

def apply_morphology(mask, kernel_size=3, operation='open'):
    """Appliquer une opération morphologique sur le masque binaire."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if operation == 'erode':
        return cv2.erode(mask, kernel, iterations=1)
    elif operation == 'dilate':
        return cv2.dilate(mask, kernel, iterations=1)
    elif operation == 'open':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        return mask

def stack_with_labels(frames, labels, font_scale=0.6):
    """Empile les frames horizontalement avec labels en dessous."""
    # Convertir en BGR si masque est en niveaux de gris
    frames_bgr = []
    for f in frames:
        if len(f.shape) == 2:  # grayscale
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        frames_bgr.append(f)

    # Redimensionner toutes les frames à la même taille
    h, w = frames_bgr[0].shape[:2]
    frames_resized = [cv2.resize(f, (w, h)) for f in frames_bgr]

    # Empile horizontalement
    stacked = np.hstack(frames_resized)

    # Ajouter une bande noire en bas pour les labels
    band_height = 40
    band = np.zeros((band_height, stacked.shape[1], 3), dtype=np.uint8)
    combined = np.vstack((stacked, band))

    # Écrire les labels
    section_width = stacked.shape[1] // len(frames)
    for i, label in enumerate(labels):
        x = i * section_width + 10
        y = h + 25
        cv2.putText(combined, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return combined

def run_mog2(video_source='seq1.mp4', history=500, varThreshold=16, detectShadows=True, morph_op='open'):
    cap = cv2.VideoCapture(video_source)
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=varThreshold,
        detectShadows=detectShadows
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        morph_mask = apply_morphology(fgmask, kernel_size=3, operation=morph_op)

        combined = stack_with_labels(
            [frame, fgmask, morph_mask],
            ["Frame originale", "Masque brut MOG2", f"Masque MOG2 + {morph_op}"]
        )

        cv2.imshow("Comparaison MOG2", combined)

        k = cv2.waitKey(30) & 0xFF
        if k == 27 or k in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

def run_gmg(video_source='seq1.mp4', initializationFrames=120, decisionThreshold=0.8, morph_op='open'):
    cap = cv2.VideoCapture(video_source)
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(
        initializationFrames=initializationFrames,
        decisionThreshold=decisionThreshold
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        morph_mask = apply_morphology(fgmask, kernel_size=3, operation=morph_op)

        combined = stack_with_labels(
            [frame, fgmask, morph_mask],
            ["Frame originale", "Masque brut GMG", f"Masque GMG + {morph_op}"]
        )

        cv2.imshow("Comparaison GMG", combined)

        k = cv2.waitKey(30) & 0xFF
        if k == 27 or k in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choisissez l'algorithme :")
    print("1 - MOG2")
    print("2 - GMG")
    choice = input("Votre choix : ")

    print("Choisissez l'opération morphologique : erode / dilate / open / close")
    morph_op = input("Opération : ").strip().lower()

    # --- Snippet to open window immediately ---
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    combined = stack_with_labels(
        [placeholder, placeholder, placeholder],
        ["Frame originale", "Masque brut", f"Masque + {morph_op}"]
    )
    cv2.imshow("Comparaison", combined)
    cv2.waitKey(1)  # forces the window to appear right away
    # ------------------------------------------

    if choice == "1":
        run_mog2('seq1.mp4', history=500, varThreshold=16, detectShadows=True, morph_op=morph_op)
    elif choice == "2":
        run_gmg('seq1.mp4', initializationFrames=120, decisionThreshold=0.8, morph_op=morph_op)
    else:
        print("Choix invalide.")

