# Verwendete Quellen
# https://github.com/raspberrypi/picamera2/blob/main/examples/opencv_face_detect.py
# https://core-electronics.com.au/guides/raspberry-pi-ai-camera-quickstart-guide/
# https://www.youtube.com/watch?v=tfp-Futa-lw
# ChatGPT


# Bibliotheken importieren
import cv2
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from HandTrackingModule_Patrick import CustomHandDetector
import os
from picamera2 import Picamera2
from PIL import Image
import smtplib
import time


# Funktion zum Versenden der eMail
def SendMail(still_image_path):
    
    # eMail Konto
    sender_name = 'Patrick Grossauer'
    sender_email = 'grossauerp1981@gmail.com'
    sender_password = 'dwrj upbq bfmp plwu'

    # Empfänger definieren
    receiver_emails = ['grossauerp@icloud.com'] # Mehrere Empfänger mit Komma getrennt

    # eMail erstellen
    msg = MIMEMultipart('related')
    msg['From'] = f'{sender_name} <{sender_email}>'
    msg['To'] = ', '.join(receiver_emails)
    msg['Subject'] = 'Person auf Terrasse ausgesperrt'

    # Textnachricht mit eingebettetem Bild erstellen
    html = '''
    <html>
    <head></head>
    <body>
    <p>An der Seestrasse 52d in 8855 Wangen SZ wurde eine Person auf der Terrasse ausgesperrt. 
    Bitte sofort Tür öffnen! Danke :)</p>
    <img src="cid:still_image" />
    </body>
    </html>
    '''
    msg.attach(MIMEText(html, 'html'))

    # Bild öffnen und als MIMEImage hinzufügen
    with open(still_image_path, 'rb') as img_file:
        img = MIMEImage(img_file.read())
        img.add_header('Content-ID', '<still_image>')  # CID, die im HTML verwendet wird
        msg.attach(img)

    # eMail über den Gmail SMTP-Server senden
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_emails, text)
        print('eMail erfolgreich gesendet!')
    except Exception as e:
        print(f'Fehler beim Senden der eMail: {e}')
    finally:
        server.quit()


# PiCamera einrichten
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 960)
picam2.preview_configuration.main.format = 'RGB888'
picam2.preview_configuration.align()
picam2.configure('preview')
picam2.start()


# Facedetector initialisieren
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cv2.startWindowThread()


# Hand Detector initialisieren (verwendet Mediapipe)
detector = CustomHandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)


# Variablen initialisieren
face_and_hands_detected = False


# Actions definieren, während PiCamera läuft
while True:
    
    # Bild der Kamera abrufen
    image = picam2.capture_array()
    
    
    ### ------ FACE DETECTOR ------
    
    # Gesichter erkennen
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.1, 5)

    # Rahmen um die erkannten Gesichter zeichnen
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 206, 209), 2)
    
    
    ### ------ HAND DETECTOR ------
    
    # Hände erkennen
    hands, image = detector.findHands(image, draw=True, flipType=True)
    
    
    ### ------ ACTIONS ------
            
    # Prüfen, ob notwendige Bedingungen für Actions erfüllt sind
    if len(faces) > 0 and len(hands) == 2:
        
        # Falls bisher die Bedingungen noch nicht erfüllt wurden, startet der Timer
        if not face_and_hands_detected:
            face_and_hands_detected_time = time.time()
            face_and_hands_detected = True
            
        # Falls während 3 Sekunden ununterbrochen die Bedingungen erfüllt sind, wird eine Nachricht versendet
        elif time.time() - face_and_hands_detected_time >= 3:
            
            # Image speichern (für die Einbettung in der eMail Nachricht)
            still_image_rgb = Image.fromarray(image[:, :, ::-1])
            still_image_path = os.path.join(os.getcwd(), 'still_images', f'still_image_{time.time()}.jpg')
            still_image_rgb.save(still_image_path)
            
            # Nachricht versenden (mit eingebettetem Bild)
            SendMail(still_image_path)
            
            # Variable zurücksetzen
            face_and_hands_detected = False
    
    # Falls die Bedingungen für Actions nicht erfüllt sind
    else:
        face_and_hands_detected = False
        
    
    # Videostream anzeigen
    cv2.imshow('Camera', image)
    
    # Videostream unterbrechen
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    
    
# PiCamera freigeben und Fenster schliessen
picam2.stop()
cv2.destroyAllWindows()