import cv2

def run_mog2(video_source='seq1.mp4', history=500, varThreshold=16, detectShadows=True):
    """Appliquer BackgroundSubtractorMOG2 sur une vidéo ou caméra."""
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

        cv2.imshow('Frame originale (MOG2)', frame)
        cv2.imshow('Masque MOG2', fgmask)

        k = cv2.waitKey(30) & 0xFF
        if k == 27 or k in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


def run_gmg(video_source='seq1.mp4', initializationFrames=120, decisionThreshold=0.8):
    """Appliquer BackgroundSubtractorGMG sur une vidéo ou caméra."""
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

        cv2.imshow('Frame originale (GMG)', frame)
        cv2.imshow('Masque GMG', fgmask)

        k = cv2.waitKey(30) & 0xFF
        if k == 27 or k in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Choisir l'algorithme à tester
    print("Choisissez l'algorithme :")
    print("1 - MOG2")
    print("2 - GMG")
    choice = input("Votre choix : ")

    if choice == "1":
        run_mog2('seq1.mp4', history=500, varThreshold=16, detectShadows=True)
    elif choice == "2":
        run_gmg('seq1.mp4', initializationFrames=120, decisionThreshold=0.8)
    else:
        print("Choix invalide.")
