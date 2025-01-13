from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re
from nltk.corpus import stopwords
import psycopg2
from psycopg2 import sql
import pyrebase
from functools import wraps
import csv
from io import StringIO
from functools import wraps

config = {
    'apiKey': 'AIzaSyBle1gLxLcBaPgLY4tPp76_ftxeag_nlwc',
    'authDomain': "classificationindobertproject.firebaseapp.com",
    'projectId': "classificationindobertproject",
    'storageBucket': "classificationindobertproject.appspot.com",
    'messagingSenderId': "935810552144",
    'appId': "1:935810552144:web:62b6449754641da6219c55",
    'databaseURL': "https://classificationindobertproject-default-rtdb.firebaseio.com"
}

config_admin = {
    'apiKey': "AIzaSyDbLSGQjod2li6iK-AZmf6KTYKdaaWlfA8",
    'authDomain': "admin-credentials-thesis.firebaseapp.com",
    'databaseURL': "https://admin-credentials-thesis-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "admin-credentials-thesis",
    'storageBucket': "admin-credentials-thesis.firebasestorage.app",
    'messagingSenderId': "308850144138",
    'appId': "1:308850144138:web:ef308dc77f23a677bcce73"
}

firebase = pyrebase.initialize_app(config)
auths = firebase.auth()

firebase_admin = pyrebase.initialize_app(config_admin)
auth_admin = firebase_admin.auth()

app = Flask(__name__)
app.secret_key = '1234'

# Define the number of classes
num_labels = 5

# Load the model and tokenizer
model_path = "C:/Users/Marcel/Documents/Skripsi/App/model/IndoBERT_Model.pth"
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased", num_labels=num_labels)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Renaming keys in the state dictionary
new_state_dict = {}
for key, value in state_dict.items():
    if key == "linear.weight":
        new_state_dict["classifier.weight"] = value
    elif key == "linear.bias":
        new_state_dict["classifier.bias"] = value
    else:
        new_state_dict[key] = value

model.load_state_dict(new_state_dict)

tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")

# Define the stop words for preprocessing
stop_words = set(stopwords.words('indonesian'))
class_names = {0: 'DS/AI/IS', 1: 'IOT', 2: 'CS/NT', 3: 'GAT/MT', 4: 'SE/MAT'}

# Preprocessing function
def cleansing(text):
    df_clean = text.lower()
    df_clean = re.sub(r"\d+", "", df_clean)
    df_clean = re.sub(r'[^\w\s]', ' ', df_clean)
    df_clean = re.sub(r'\s+', ' ', df_clean)
    df_clean = re.sub(r'\#\S*', '', df_clean)
    df_clean = ' '.join(word for word in df_clean.split() if word not in stop_words)
    return df_clean

# Prediction function
def predict(text):
    cleansed_text = cleansing(text)
    inputs = tokenizer(cleansed_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()  # Convert logits to probabilities
    prediction = torch.argmax(logits, dim=1).item()
    result = class_names[prediction]
    return result, probabilities


# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect("postgresql://postgres:200403@localhost:5432/thesis_db")
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

# Decorator to check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('authenticate'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'admin':
            return render_template('403.html'), 403
        return f(*args, **kwargs)
    return decorated_function

# Define the route for the home page (predictor)
@app.route('/')
@app.route('/authenticate', methods=['GET', 'POST'])
def authenticate():
    if request.method == 'POST':
        email = request.form['user_email']
        password = request.form['user_pwd']
        try:
            user_info = auths.sign_in_with_email_and_password(email, password)
            account_info = auths.get_account_info(user_info['idToken'])
            if not account_info['users'][0]['emailVerified']:
                verify_message = 'Please verify your email'
                return render_template('authenticate.html', umessage=verify_message)
            session['user'] = email  # Store user email in session
            session['role'] = 'student'
            return render_template('index.html')
        except Exception:
            unsuccessful = 'Please check your credentials'
            return render_template('authenticate.html', umessage=unsuccessful)
    return render_template('authenticate.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def authenticate_admin():
    if request.method == 'POST':
        email = request.form['user_email']
        password = request.form['user_pwd']
        try:
            user_info = auth_admin.sign_in_with_email_and_password(email, password)
            account_info = auth_admin.get_account_info(user_info['idToken'])
            if not account_info['users'][0]['emailVerified']:
                verify_message = 'Please verify your email'
                return render_template('admin_login.html', umessage=verify_message)
            session['user'] = email  # Store user email in session
            session['role'] = 'admin'  # Store user role in session
            return render_template('index_admin.html')
        except Exception:
            unsuccessful = 'Please check your credentials'
            return render_template('admin_login.html', umessage=unsuccessful)
    return render_template('admin_login.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict_route():
    text = request.form['text']
    prediction, probabilities = predict(text)
    response = {
        'class': prediction,
        'probabilities': probabilities
    }
    return jsonify(response)


# Define the route for fetching slot data
@login_required
@app.route('/get_slot', methods=['POST'])
def get_slot():
    supervisor_id = request.form['supervisor_id']

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = sql.SQL("SELECT slot FROM supervisor_data WHERE supervisor_id = %s")
        cur.execute(query, (supervisor_id,))
        slot = cur.fetchone()[0]
        cur.close()
        conn.close()
        return jsonify({'slot': slot})
    except Exception as e:
        print(f"Error fetching slot data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Define the route for saving data
@login_required
@app.route('/save', methods=['POST'])
def save_data():
    # Mengambil data dari form
    title = request.form['title']
    abstract = request.form['abstract']
    supervisor = request.form['supervisor']
    supervisor_id = request.form.get('supervisor_id')
    topic = request.form.get('topic')

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Ambil student_id dari tabel student_information berdasarkan email pengguna yang sedang login
        user_email = session.get('user')  # Mengambil email dari session
        cur.execute("SELECT student_id FROM student_information WHERE email = %s", (user_email,))
        result = cur.fetchone()

        if result is None:
            return jsonify({'status': 'error', 'message': 'Student ID not found for the logged-in user.'}), 400
        
        student_id = result[0]  # Mengambil student_id dari hasil query

        # Cek apakah pengguna memiliki data dengan status "Pending" di thesis_data
        cur.execute("SELECT id FROM thesis_data WHERE student_id = %s AND status = 'Pending'", (student_id,))
        pending_data = cur.fetchone()

        if pending_data:
            return jsonify({
                'status': 'error', 
                'message': 'You already have a thesis submission with status "Pending". Please wait for approval or rejection before submitting again.'
            }), 400

        # Simpan data ke tabel thesis_data dengan status default "Pending"
        query = sql.SQL(
            "INSERT INTO thesis_data (title, abstract, supervisor, supervisor_id, topic, student_id, status) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        cur.execute(query, (title, abstract, supervisor, supervisor_id, topic, student_id, 'Pending'))
        
        # Jika supervisor_id diberikan, kurangi slot supervisor
        if supervisor_id:
            update_slot_query = sql.SQL("UPDATE supervisor_data SET slot = slot - 1 WHERE supervisor_id = %s")
            cur.execute(update_slot_query, (supervisor_id,))
        
        # Commit perubahan
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

# Define the route for saving data
@login_required
@admin_required
@app.route('/save_admin_mode', methods=['POST'])
def save_data_admin_mode():
    name = request.form['name']
    nim = request.form['nim']
    title = request.form['title']
    abstract = request.form['abstract']
    supervisor = request.form['supervisor']
    supervisor_id = request.form.get('supervisor_id')
    topic = request.form.get('topic')

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if NIM already exists
        check_student_query = sql.SQL("SELECT COUNT(*) FROM student_information WHERE student_id = %s")
        cur.execute(check_student_query, (nim,))
        student_exists = cur.fetchone()[0] > 0
        
        if not student_exists:
            # Save student information if NIM does not exist
            student_query = sql.SQL("INSERT INTO student_information (student_id, name) VALUES (%s, %s)")
            cur.execute(student_query, (nim, name))
        
        # Save thesis data
        thesis_query = sql.SQL("INSERT INTO thesis_data (title, abstract, supervisor, supervisor_id, topic) VALUES (%s, %s, %s, %s, %s)")
        cur.execute(thesis_query, (title, abstract, supervisor, supervisor_id, topic))
        
        if supervisor_id:
            update_slot_query = sql.SQL("UPDATE supervisor_data SET slot = slot - 1 WHERE supervisor_id = %s")
            cur.execute(update_slot_query, (supervisor_id,))
        
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


    
@app.route('/result_status', methods=['GET'])
@login_required
def result_status():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Ambil student_id dari tabel student_information berdasarkan email pengguna
        user_email = session.get('user')  # Email pengguna dari session
        cur.execute("SELECT student_id FROM student_information WHERE email = %s", (user_email,))
        result = cur.fetchone()

        if result is None:
            return "Student ID not found for the logged-in user.", 404
        
        student_id = result[0]  # Ambil student_id

        # Ambil data dari thesis_data berdasarkan student_id
        cur.execute("""
            SELECT id, title, abstract, supervisor, supervisor_id, topic, status, reason
            FROM thesis_data
            WHERE student_id = %s
        """, (student_id,))
        rows = cur.fetchall()

        cur.close()
        conn.close()

        # Render HTML dan kirim data
        return render_template('result_status.html', rows=rows)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching data.", 500

@login_required
@admin_required
@app.route('/check_pending_status', methods=['GET'])
def check_pending_status():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Ambil student_id dari tabel student_information berdasarkan email pengguna yang sedang login
        user_email = session.get('user')  # Mengambil email dari session
        cur.execute("SELECT student_id FROM student_information WHERE email = %s", (user_email,))
        result = cur.fetchone()

        if result is None:
            return jsonify({'status': 'error', 'message': 'Student ID not found for the logged-in user.'}), 400
        
        student_id = result[0]  # Mengambil student_id dari hasil query

        # Cek apakah pengguna memiliki data dengan status "Pending" di thesis_data
        cur.execute("SELECT id FROM thesis_data WHERE student_id = %s AND status = 'Pending'", (student_id,))
        pending_data = cur.fetchone()

        cur.close()
        conn.close()

        if pending_data:
            return jsonify({'status': 'pending'})
        else:
            return jsonify({'status': 'no_pending'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Define the route for displaying results
@app.route('/approval_page')
@login_required
@admin_required
def approval():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM thesis_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('approval_page.html', rows=rows)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching results."
    
@app.route('/results')
@login_required
@admin_required
def results():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM thesis_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('results.html', rows=rows)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching results."

@app.route('/homepage', methods=['GET'])
@login_required
def homepage():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching supervisor list."

@app.route('/homepage_admin', methods=['GET'])
@login_required
@admin_required
def homepage_admin():
    try:
        return render_template('index_admin.html')
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching supervisor list."

@app.route('/spv_list_admin')
@login_required
@admin_required
def spv_list_admin():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM supervisor_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('spv_list_admin.html', rows=rows)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching supervisor list."

@app.route('/update_slot_admin', methods=['POST'])
@login_required
@admin_required
def update_slot_admin():
    try:
        data = request.get_json()
        supervisor_id = data['id']
        slot = data['slot']
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE supervisor_data SET slot = %s WHERE supervisor_id = %s", (slot, supervisor_id))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

    
# Define the route for displaying the supervisor list
@app.route('/supervisor_list')
@login_required
def supervisor_list():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM supervisor_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('supervisor_list.html', rows=rows)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while fetching supervisor list."

@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        name = request.form['name']
        student_id = request.form['student_id']
        email = request.form['user_email']
        password = request.form['user_pwd0']
        pwd0 = request.form['user_pwd0']
        pwd1 = request.form['user_pwd1']
        
        if pwd0 == pwd1:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                
                # Check if student_id already exists
                check_student_query = sql.SQL("SELECT COUNT(*) FROM student_information WHERE student_id = %s")
                cur.execute(check_student_query, (student_id,))
                student_exists = cur.fetchone()[0] > 0
                
                if student_exists:
                    # Update existing student information
                    update_student_query = sql.SQL(
                        "UPDATE student_information SET name = %s, email = %s, password = %s WHERE student_id = %s"
                    )
                    cur.execute(update_student_query, (name, email, password, student_id))
                else:
                    # Insert new student information
                    insert_student_query = sql.SQL(
                        "INSERT INTO student_information (name, student_id, email, password) VALUES (%s, %s, %s, %s)"
                    )
                    cur.execute(insert_student_query, (name, student_id, email, password))
                
                conn.commit()
                cur.close()
                conn.close()
                
                new_user = auths.create_user_with_email_and_password(email, password)
                auths.send_email_verification(new_user['idToken'])
                return render_template('verify_email.html')
            except Exception as e:
                conn.rollback()
                existing_account = 'This email is already used'
                return render_template('create_account.html', exist_message=existing_account)
        else:
            mismatch_message = 'Passwords do not match'
            return render_template('create_account.html', exist_message=mismatch_message)
    
    return render_template('create_account.html')

@app.route("/reset_password", methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['user_email']
        auths.send_password_reset_email(email)
        return render_template('authenticate.html')
    return render_template('reset_password.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)  # Remove user from session
    return redirect(url_for('authenticate'))

# Route for generating CSV
@app.route('/generate_csv')
@login_required
def generate_csv():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM thesis_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Create a string buffer to hold CSV data
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)

        # Write header
        writer.writerow(['No', 'Title', 'Abstract', 'Supervisor', 'SpvID', 'Topic', 'Status', 'Reason', 'NIM'])

        # Write data
        for row in rows:
            writer.writerow(row)

        # Get the CSV string from the buffer
        csv_buffer.seek(0)
        csv_data = csv_buffer.getvalue()

        # Create a response with the CSV data
        response = Response(csv_data, mimetype='text/csv')
        response.headers.set("Content-Disposition", "attachment", filename="thesis_data.csv")
        return response
    except Exception as e:
        print(f"Error generating CSV: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
# Route to generate and download CSV
@app.route('/generate_supervisor_csv')
@login_required
def generate_supervisor_csv():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM supervisor_data")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Kode Dosen', 'Nama Dosen', 'Slot'])  # Header row
        writer.writerows(rows)
        output.seek(0)

        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=supervisor_list.csv"}
        )
    except Exception as e:
        print(f"Error generating CSV: {e}")
        return "An error occurred while generating the CSV.", 500

@app.route('/update_status', methods=['POST'])
def update_status():
    try:
        thesis_id = request.json.get('id')
        new_status = request.json.get('status')
        
        # Debugging log
        print(f"Received ID: {thesis_id}, Status: {new_status}")

        conn = get_db_connection()
        cur = conn.cursor()
        query = sql.SQL("UPDATE thesis_data SET status = %s WHERE id = %s")
        cur.execute(query, (new_status, thesis_id))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'status': 'success', 'new_status': new_status})
    except Exception as e:
        print(f"Error updating status: {e}")  # Log error to the console
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_slot', methods=['POST'])
def update_slot():
    try:
        # Ambil data dari request
        thesis_id = request.json.get('id')
        reason = request.json.get('reason')  # Alasan penolakan

        # Debugging log
        print(f"Received Thesis ID: {thesis_id}, Reason: {reason}")

        # Koneksi ke database
        conn = get_db_connection()
        cur = conn.cursor()

        # Perbarui reason di tabel thesis_data
        cur.execute("UPDATE thesis_data SET reason = %s, status = 'Rejected' WHERE id = %s", (reason, thesis_id))
        print(f"Reason updated and status set to Rejected for Thesis ID: {thesis_id}")

        # Ambil supervisor_id dari thesis_data
        cur.execute("SELECT supervisor_id FROM thesis_data WHERE id = %s", (thesis_id,))
        supervisor_data = cur.fetchone()

        if supervisor_data is None or supervisor_data[0] is None:
            print(f"Supervisor ID for Thesis ID {thesis_id} is NULL.")
            return jsonify({'status': 'error', 'message': 'Supervisor ID is NULL.'}), 404

        supervisor_id = supervisor_data[0]
        print(f"Supervisor ID: {supervisor_id}")

        # Tambahkan slot supervisor di tabel supervisor_data
        cur.execute("UPDATE supervisor_data SET slot = slot + 1 WHERE supervisor_id = %s", (supervisor_id,))
        print(f"Slot incremented for Supervisor ID: {supervisor_id}")

        # Commit perubahan ke database
        conn.commit()
        print("Database commit successful.")

        # Tutup koneksi database
        cur.close()
        conn.close()

        # Kirim response sukses
        return jsonify({'status': 'success', 'message': 'Slot updated successfully and status set to Rejected.'})
    except Exception as e:
        print(f"Error updating slot: {e}")  # Log error ke konsol
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/update_reason', methods=['POST'])
def update_reason():
    try:
        data = request.json
        thesis_id = data.get('id')
        reason = data.get('reason')

        # Update reason dan status di database
        conn = get_db_connection()
        cur = conn.cursor()
        query = sql.SQL("UPDATE thesis_data SET status = %s, reason = %s WHERE id = %s")
        cur.execute(query, ('Rejected', reason, thesis_id))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error updating reason: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
