from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify, flash
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import openpyxl
import os
import sqlite3
import razorpay
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'emotiondetection_secret_2025'

# ================= RAZORPAY CONFIG =================
# ✅ Apni keys yahan rakho
RAZORPAY_KEY_ID     = 'rzp_test_SrCLYW4vCOEYVr'       # Replace karo
RAZORPAY_KEY_SECRET = 'n6t99tNfDtHR6IEoFcxADA2G'             # Replace karo

razorpay_client = razorpay.Client(
    auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
)

FREE_DOWNLOADS   = 2       # kitne free downloads
PRICE_PER_DOWNLOAD = 2900  # Rs 29 in paise (Razorpay uses paise)

# ================= FOLDERS =================
PROFILE_FOLDER = os.path.join('static', 'uploads', 'profiles')
UPLOAD_FOLDER  = os.path.join('static', 'uploads')
os.makedirs(PROFILE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name     TEXT NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            mobile        TEXT DEFAULT '',
            password      TEXT,
            provider      TEXT DEFAULT 'email',
            photo         TEXT DEFAULT '',
            download_count INTEGER DEFAULT 0,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Add columns if old DB
    for col in ['photo TEXT DEFAULT ""', 'mobile TEXT DEFAULT ""',
                'download_count INTEGER DEFAULT 0']:
        try:
            c.execute(f'ALTER TABLE users ADD COLUMN {col}')
        except:
            pass

    # Payments table
    c.execute('''
        CREATE TABLE IF NOT EXISTS payments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER,
            order_id      TEXT,
            payment_id    TEXT,
            amount        INTEGER,
            status        TEXT DEFAULT 'created',
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ================= SAVE USER DATA =================
def save_user_data(name, mobile, email):
    file = "users.xlsx"
    if not os.path.exists(file):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Name", "Mobile", "Email"])
        wb.save(file)
    wb = openpyxl.load_workbook(file)
    wb.active.append([name, mobile, email])
    wb.save(file)

# ================= LOAD MODEL =================
with open("emotiondetector.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("emotiondetector.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

labels = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}
emotion_colors = {
    'angry':(0,0,255), 'disgust':(0,140,0), 'fear':(128,0,128),
    'happy':(0,215,255), 'neutral':(255,255,255),
    'sad':(255,100,0), 'surprise':(0,165,255),
}

emotion_buffer = []
stable_label   = "neutral"
stable_conf    = 0

# ================= HELPERS =================
def extract_features(image):
    return np.array(image).reshape(1, 48, 48, 1) / 255.0

def get_user_by_email(email):
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_id(uid):
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (uid,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(full_name, email, password, mobile='', provider='email', photo=''):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed = generate_password_hash(password) if password else None
    c.execute(
        'INSERT INTO users (full_name, email, mobile, password, provider, photo) VALUES (?,?,?,?,?,?)',
        (full_name, email, mobile, hashed, provider, photo)
    )
    conn.commit()
    uid = c.lastrowid
    conn.close()
    return uid

def update_password(email, new_password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('UPDATE users SET password=? WHERE email=?',
              (generate_password_hash(new_password), email))
    conn.commit()
    conn.close()

def increment_download(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('UPDATE users SET download_count = download_count + 1 WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()

def get_download_count(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT download_count FROM users WHERE id = ?', (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

def save_payment(user_id, order_id, amount):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO payments (user_id, order_id, amount) VALUES (?,?,?)',
              (user_id, order_id, amount))
    conn.commit()
    conn.close()

def mark_payment_success(order_id, payment_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('UPDATE payments SET status=?, payment_id=? WHERE order_id=?',
              ('paid', payment_id, order_id))
    conn.commit()
    conn.close()

def save_profile_photo(photo_file, email):
    if not photo_file or photo_file.filename == '':
        return ''
    ext      = photo_file.filename.rsplit('.', 1)[-1].lower()
    filename = secure_filename(email.replace('@','_').replace('.','_')) + '.' + ext
    filepath = os.path.join(PROFILE_FOLDER, filename)
    photo_file.save(filepath)
    return 'uploads/profiles/' + filename

def set_session(user_id, name, email, photo):
    session['user_id']    = user_id
    session['user_name']  = name
    session['user_email'] = email
    session['user_photo'] = photo or ''

# ================= CAMERA =================
def generate_frames():
    global emotion_buffer, stable_label, stable_conf
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = camera.read()
        if not success:
            break
        gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(80,80))

        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            face  = cv2.resize(gray[y:y+h, x:x+w], (48,48))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            face  = clahe.apply(face)
            pred  = model.predict(extract_features(face), verbose=0)[0]

            emotion_buffer.append(pred)
            if len(emotion_buffer) > 5:
                emotion_buffer.pop(0)

            avg  = np.mean(emotion_buffer, axis=0)
            lbl  = labels[avg.argmax()]
            conf = int(avg.max() * 100)

            if conf >= 40:
                stable_label = lbl
                stable_conf  = conf

            color = emotion_colors.get(stable_label, (0,255,0))
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f'{stable_label.upper()} {stable_conf}%',
                        (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            emotion_buffer.clear()

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= ROUTES =================

@app.route('/')
def index():
    return render_template('index.html')

# -------- LOGIN --------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if 'user_id' in session:
            return redirect(url_for('camera_page'))
        return render_template('login.html')

    if request.is_json:
        data     = request.get_json()
        email    = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'success':False, 'message':'❌ Email aur Password bharo!'})

        user = get_user_by_email(email)
        if not user:
            return jsonify({'success':False, 'message':'❌ Email registered nahi hai!'})
        if not user['password'] or not check_password_hash(user['password'], password):
            return jsonify({'success':False, 'message':'❌ Password galat hai!'})

        set_session(user['id'], user['full_name'], user['email'], user['photo'])
        return jsonify({'success':True, 'message':f'✅ Welcome {user["full_name"]}!', 'redirect':'/camera'})

    email    = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    user     = get_user_by_email(email)
    if not user or not user['password'] or not check_password_hash(user['password'], password):
        flash('❌ Email ya password galat hai!')
        return redirect(url_for('login'))
    set_session(user['id'], user['full_name'], user['email'], user['photo'])
    return redirect(url_for('camera_page'))

# -------- SIGNUP --------
@app.route('/signup', methods=['POST'])
def signup():
    try:
        name    = request.form.get('full_name', '').strip()
        email   = request.form.get('email', '').strip().lower()
        mobile  = request.form.get('mobile', '').strip()
        passw   = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        photo_f = request.files.get('photo')

        if not name and request.is_json:
            data    = request.get_json()
            name    = data.get('full_name','').strip()
            email   = data.get('email','').strip().lower()
            mobile  = data.get('mobile','').strip()
            passw   = data.get('password','')
            confirm = data.get('confirm_password','')
            photo_f = None

        if not name or not email or not passw:
            return jsonify({'success':False, 'message':'❌ Sabhi fields fill karo!'})
        if len(passw) < 6:
            return jsonify({'success':False, 'message':'❌ Password kam se kam 6 characters!'})
        if passw != confirm:
            return jsonify({'success':False, 'message':'❌ Passwords match nahi!'})
        if get_user_by_email(email):
            return jsonify({'success':False, 'message':'❌ Email already registered!'})

        photo_path = save_profile_photo(photo_f, email) if photo_f else ''
        create_user(name, email, passw, mobile, 'email', photo_path)
        return jsonify({'success':True, 'message':f'🎉 Account ban gaya {name}!', 'redirect':'/login'})

    except Exception as e:
        return jsonify({'success':False, 'message':f'❌ Server error: {str(e)}'})

# -------- REGISTER PAGE --------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name    = request.form.get('name','').strip()
        email   = request.form.get('email','').strip().lower()
        mobile  = request.form.get('mobile','').strip()
        password= request.form.get('password','')
        photo_f = request.files.get('photo')
        if get_user_by_email(email):
            return jsonify({'success':False, 'message':'❌ Email already registered!'})
        photo_path = save_profile_photo(photo_f, email) if photo_f else ''
        create_user(name, email, password, mobile, 'email', photo_path)
        return redirect(url_for('login'))
    return render_template('register.html')

# -------- FORGOT / RESET PASSWORD --------
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    try:
        data  = request.get_json()
        email = data.get('email','').strip().lower()
        if not email:
            return jsonify({'success':False, 'message':'❌ Email bharo!'})
        if not get_user_by_email(email):
            return jsonify({'success':False, 'message':'❌ Email registered nahi hai!'})
        return jsonify({'success':True, 'message':'✅ OTP Ready! Demo: 123456'})
    except Exception as e:
        return jsonify({'success':False, 'message':f'❌ Error: {str(e)}'})

@app.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        data     = request.get_json()
        email    = data.get('email','').strip().lower()
        password = data.get('password','')
        confirm  = data.get('confirm','')
        if not email or not password:
            return jsonify({'success':False, 'message':'❌ Fields missing!'})
        if len(password) < 6:
            return jsonify({'success':False, 'message':'❌ Password 6+ characters!'})
        if password != confirm:
            return jsonify({'success':False, 'message':'❌ Passwords match nahi!'})
        if not get_user_by_email(email):
            return jsonify({'success':False, 'message':'❌ Email not found!'})
        update_password(email, password)
        return jsonify({'success':True, 'message':'✅ Password reset ho gaya!', 'redirect':'/login'})
    except Exception as e:
        return jsonify({'success':False, 'message':f'❌ Error: {str(e)}'})

@app.route('/reset')
def reset():
    return render_template('reset.html')

# -------- LOGOUT --------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# -------- CAMERA --------
@app.route('/camera')
def camera_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('camera.html')

# -------- PROFILE --------
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = get_user_by_email(session['user_email'])
    return render_template('profile.html', user=user)

# ================= PAYMENT ROUTES =================

# -------- CHECK DOWNLOAD STATUS --------
@app.route('/check-download', methods=['GET'])
def check_download():
    if 'user_id' not in session:
        return jsonify({'success':False, 'message':'Login karo!'})

    user_id = session['user_id']
    count   = get_download_count(user_id)
    remaining_free = max(0, FREE_DOWNLOADS - count)

    return jsonify({
        'success':        True,
        'download_count': count,
        'free_downloads': FREE_DOWNLOADS,
        'remaining_free': remaining_free,
        'needs_payment':  count >= FREE_DOWNLOADS,
        'price':          29
    })

# -------- CREATE RAZORPAY ORDER --------
@app.route('/create-order', methods=['POST'])
def create_order():
    if 'user_id' not in session:
        return jsonify({'success':False, 'message':'Login karo!'})

    try:
        order = razorpay_client.order.create({
            'amount':   PRICE_PER_DOWNLOAD,
            'currency': 'INR',
            'payment_capture': 1
        })

        save_payment(session['user_id'], order['id'], PRICE_PER_DOWNLOAD)

        return jsonify({
            'success':    True,
            'order_id':   order['id'],
            'amount':     PRICE_PER_DOWNLOAD,
            'currency':   'INR',
            'key_id':     RAZORPAY_KEY_ID,
            'user_name':  session.get('user_name',''),
            'user_email': session.get('user_email','')
        })

    except Exception as e:
        return jsonify({'success':False, 'message':f'❌ Order create nahi hua: {str(e)}'})

# -------- VERIFY PAYMENT & ALLOW DOWNLOAD --------
@app.route('/verify-payment', methods=['POST'])
def verify_payment():
    if 'user_id' not in session:
        return jsonify({'success':False, 'message':'Login karo!'})

    try:
        data       = request.get_json()
        order_id   = data.get('razorpay_order_id')
        payment_id = data.get('razorpay_payment_id')
        signature  = data.get('razorpay_signature')

        # Verify signature
        razorpay_client.utility.verify_payment_signature({
            'razorpay_order_id':   order_id,
            'razorpay_payment_id': payment_id,
            'razorpay_signature':  signature
        })

        # Payment verified — allow download
        mark_payment_success(order_id, payment_id)
        increment_download(session['user_id'])

        return jsonify({'success':True, 'message':'✅ Payment successful! Download shuru karo.'})

    except Exception as e:
        return jsonify({'success':False, 'message':f'❌ Payment verify nahi hua: {str(e)}'})

# -------- FREE DOWNLOAD (first 2) --------
@app.route('/record-download', methods=['POST'])
def record_download():
    if 'user_id' not in session:
        return jsonify({'success':False, 'message':'Login karo!'})

    user_id = session['user_id']
    count   = get_download_count(user_id)

    if count < FREE_DOWNLOADS:
        increment_download(user_id)
        return jsonify({'success':True, 'message':'✅ Free download!', 'remaining': FREE_DOWNLOADS - count - 1})
    else:
        return jsonify({'success':False, 'message':'❌ Free downloads khatam!', 'needs_payment': True})

# ================= OTHER ROUTES =================

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current-emotion')
def current_emotion():
    return jsonify({'emotion':stable_label, 'confidence':stable_conf})

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error':'Image nahi mili!'}), 400
    file      = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error':'Image read error!'}), 400
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(40,40))
    if len(faces) == 0:
        return jsonify({'success':False, 'error':'Koi face detect nahi hua!'})
    faces    = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x,y,w,h  = faces[0]
    face     = cv2.resize(gray[y:y+h, x:x+w], (48,48))
    pred     = model.predict(extract_features(face), verbose=0)[0]
    dominant = labels[pred.argmax()]
    emotions = {labels[i]: round(float(pred[i]),4) for i in range(len(labels))}
    return jsonify({'success':True, 'dominant_emotion':dominant,
                    'emotions':emotions, 'confidence':round(float(pred.max()),4),
                    'face_count':len(faces)})

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('result.html', emotion='no face detected', image='')
    file = request.files['image']
    if file.filename == '':
        return render_template('result.html', emotion='no face detected', image='')
    safe_name = secure_filename(file.filename)
    filepath  = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)
    img = cv2.imread(filepath)
    if img is None:
        return render_template('result.html', emotion='no face detected', image='')
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    label = 'no face detected'
    for (x,y,w,h) in faces:
        face  = cv2.resize(gray[y:y+h, x:x+w], (48,48))
        pred  = model.predict(extract_features(face), verbose=0)
        label = labels[pred.argmax()]
    return render_template('result.html', emotion=label, image='uploads/'+safe_name)

@app.route('/submit', methods=['POST'])
def submit():
    save_user_data(request.form['name'], request.form['mobile'], request.form['email'])
    return redirect(url_for('camera_page'))

@app.route('/upload-page')
def upload_page():
    return render_template('upload.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/face-detection')
def face_detection():
    return render_template('face_detection.html')

@app.route('/education')
def education():
    return render_template('education.html')

@app.route('/healthcare')
def healthcare():
    return render_template('healthcare.html')

@app.route('/security')
def security():
    return render_template('security.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/student-engagement')
def student_engagement():
    return render_template('student_engagement.html')

@app.route('/focus-detection')
def focus_detection():
    return render_template('focus_detection.html')

@app.route('/smart-learning')
def smart_learning():
    return render_template('smart_learning.html')

@app.route('/live-detection')
def live_detection():
    return render_template('live_detection.html')

@app.route('/image-upload')
def image_upload():
    return render_template('upload.html')

@app.route('/rest-api')
def rest_api():
    return render_template('documentation.html')

@app.route('/auth/google', methods=['POST'])
def google_auth():
    try:
        data  = request.get_json()
        email = data.get('email','').strip().lower()
        name  = data.get('name','Google User')
        if not email:
            return jsonify({'success':False, 'message':'❌ Email required!'})
        user = get_user_by_email(email)
        uid  = create_user(name, email, None, provider='google') if not user else user['id']
        photo = user['photo'] if user else ''
        set_session(uid, name, email, photo)
        return jsonify({'success':True, 'message':f'✅ Welcome {name}!', 'redirect':'/camera'})
    except Exception as e:
        return jsonify({'success':False, 'message':f'❌ Error: {str(e)}'})

@app.route('/auth/facebook', methods=['POST'])
def facebook_auth():
    try:
        data  = request.get_json()
        email = data.get('email','').strip().lower()
        name  = data.get('name','Facebook User')
        if not email:
            return jsonify({'success':False, 'message':'❌ Email required!'})
        user = get_user_by_email(email)
        uid  = create_user(name, email, None, provider='facebook') if not user else user['id']
        photo = user['photo'] if user else ''
        set_session(uid, name, email, photo)
        return jsonify({'success':True, 'message':f'✅ Welcome {name}!', 'redirect':'/camera'})
    except Exception as e:
        return jsonify({'success':False, 'message':f'❌ Error: {str(e)}'})
    
#..............terms...............
@app.route('/terms')
def terms():
    return render_template('terms.html')


if __name__ == "__main__":
    app.run(debug=True)