from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import logging
from datetime import datetime
from bson import ObjectId


def _json_safe(obj):
    """Convert object to JSON-serializable format."""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def create_app():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app = Flask(__name__, static_folder=os.path.join(base_dir, 'static'), static_url_path='/static')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    app.logger = logging.getLogger('agri_scan')
    app.logger.info("Starting AgriScan application")

    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    app.config['UPLOAD_FOLDER'] = 'data/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['MONGO_URI'] = os.environ.get('MONGO_URI', 'mongodb+srv://Rinae-User:RamadiRR@cluster1.yzmbbvs.mongodb.net/agriscan?appName=Cluster1')
    app.config['BCRYPT_LOG_ROUNDS'] = 12

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    bcrypt = None
    try:
        from flask_bcrypt import Bcrypt
        bcrypt = Bcrypt(app)
    except Exception as e:
        print(f"Could not initialize Bcrypt: {e}")

    mongo_service = None
    try:
        from app.services.mongo_service import MongoService
        mongo_service = MongoService()
    except Exception as e:
        print(f"Could not initialize MongoDB service: {e}")

    try:
        from app.services.agriculture_ml_service import AgricultureMLService
        ml_service = AgricultureMLService()
        print("Agriculture ML Service initialized successfully!")
    except Exception as e:
        print(f"Could not initialize ML service: {e}")
        ml_service = None

    def hash_password(password):
        if bcrypt:
            return bcrypt.generate_password_hash(password).decode('utf-8')
        return password

    def check_password(password, hashed):
        if bcrypt:
            return bcrypt.check_password_hash(hashed, password)
        return password == hashed

    def log_action(action, details=None, user_id=None):
        if mongo_service:
            mongo_service.insert_log({
                'action': action,
                'details': details or '',
                'user_id': user_id or session.get('user_id'),
                'user_email': session.get('user_email'),
                'user_role': session.get('user_role')
            })

    @app.route('/')
    def index():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('user_role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return render_template('index.html', user=session)

    @app.route('/login')
    def login():
        if 'user_id' in session:
            if session.get('user_role') == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('index'))
        return render_template('login.html')

    @app.route('/login', methods=['POST'])
    def login_post():
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        user = mongo_service.find_user_by_email(email) if mongo_service else None
        if user and check_password(password, user.get('password', '')):
            session['user_id'] = user['id']
            session['user_name'] = user.get('firstName', '') + ' ' + user.get('lastName', '')
            session['user_email'] = email
            session['user_role'] = user.get('role', 'user')
            log_action('login', f"User {email} logged in", user['id'])
            return jsonify({'success': True, 'message': 'Login successful', 'role': user.get('role', 'user')})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    @app.route('/register')
    def register():
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('register.html')

    @app.route('/register', methods=['POST'])
    def register_post():
        data = request.get_json()

        required_fields = ['firstName', 'lastName', 'email', 'password', 'confirmPassword', 'phone', 'location']
        for field in required_fields:
            if not data.get(field, '').strip():
                return jsonify({'success': False, 'message': f'{field} is required'}), 400

        email = data['email'].strip().lower()
        password = data['password']
        confirm_password = data['confirmPassword']

        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'}), 400

        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters long'}), 400

        if mongo_service and mongo_service.find_user_by_email(email):
            return jsonify({'success': False, 'message': 'Email already registered'}), 409

        user_data = {
            'firstName': data['firstName'].strip(),
            'lastName': data['lastName'].strip(),
            'email': email,
            'password': hash_password(password),
            'phone': data['phone'].strip(),
            'location': data['location'],
            'farmSize': data.get('farmSize', ''),
            'role': 'user'
        }

        if mongo_service:
            user_id = mongo_service.create_user(user_data)
            if user_id:
                log_action('register', f"New user registered: {email}", user_id)
                return jsonify({'success': True, 'message': 'Registration successful'})
            else:
                return jsonify({'success': False, 'message': 'Failed to create account'}), 500
        else:
            return jsonify({'success': False, 'message': 'Database not available'}), 500

    @app.route('/logout')
    def logout():
        user_email = session.get('user_email')
        log_action('logout', f"User {user_email} logged out")
        session.clear()
        return redirect(url_for('login'))

    @app.route('/plant-scan')
    def plant_scan():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('user_role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return render_template('plant_scan.html', user=session)

    @app.route('/api/analyze-plant', methods=['POST'])
    def analyze_plant():
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400

        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload JPG, PNG, or WebP images.'}), 400

        try:
            from PIL import Image
            import io

            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = image.resize((224, 224))

            result = ml_service.analyze_plant_image(image) if ml_service else {
                'success': True,
                'disease': 'Healthy Plant',
                'confidence': 95.0,
                'description': 'Your plant appears to be healthy with no visible signs of disease.',
                'recommendations': [
                    'Continue regular watering and fertilization',
                    'Monitor for any changes in leaf color or texture',
                    'Maintain proper spacing between plants for air circulation'
                ]
            }

            result['timestamp'] = datetime.now().isoformat()
            result['filename'] = file.filename
            result['user_id'] = session.get('user_id')

            if mongo_service:
                try:
                    scan_id = mongo_service.insert_scan(result)
                    if scan_id:
                        result['scan_id'] = scan_id
                        log_action('plant_scan', f"Disease detected: {result.get('disease')}", session.get('user_id'))
                except Exception as e:
                    print(f"Failed to save scan to MongoDB: {e}")

            return jsonify(_json_safe(result))

        except Exception as e:
            return jsonify({'success': False, 'error': f'Error analyzing image: {str(e)}'}), 500

    @app.route('/api/scans')
    def get_scans():
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Authentication required'}), 401

        user_role = session.get('user_role')
        limit = int(request.args.get('limit', 50))
        skip = int(request.args.get('skip', 0))
        disease_filter = request.args.get('disease')

        if mongo_service is None:
            return jsonify({'success': False, 'error': 'Database not connected'}), 500

        scans = mongo_service.find_scans(limit=limit, skip=skip, disease_filter=disease_filter)
        total = mongo_service.count_scans(disease_filter=disease_filter)
        return jsonify({'success': True, 'scans': scans, 'total': total})

    @app.route('/api/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'app_name': 'AgriScan',
            'ml_service': 'active' if ml_service else 'unavailable',
            'mongodb': 'connected' if mongo_service and mongo_service.db is not None else 'disconnected',
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/admin')
    def admin_dashboard():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('user_role') != 'admin':
            return redirect(url_for('index'))
        return render_template('admin/dashboard.html', user=session)

    @app.route('/admin/scans')
    def admin_scans():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return redirect(url_for('index'))
        return render_template('admin/scans.html', user=session)

    @app.route('/admin/users')
    def admin_users():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return redirect(url_for('index'))
        return render_template('admin/users.html', user=session)

    @app.route('/admin/diseases')
    def admin_diseases():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return redirect(url_for('index'))
        return render_template('admin/diseases.html', user=session)

    @app.route('/admin/logs')
    def admin_logs():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return redirect(url_for('index'))
        return render_template('admin/logs.html', user=session)

    @app.route('/admin/api/analytics')
    def admin_analytics_api():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        if mongo_service is None:
            return jsonify({'success': False, 'error': 'Database not connected'}), 500

        analytics = mongo_service.get_analytics_summary()
        return jsonify({'success': True, 'data': analytics})

    @app.route('/admin/api/users')
    def admin_users_api():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        if mongo_service is None:
            return jsonify({'success': False, 'error': 'Database not connected'}), 500

        users = mongo_service.find_users(limit=100)
        return jsonify({'success': True, 'users': users})

    @app.route('/admin/api/users', methods=['POST'])
    def admin_create_user_api():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        data = request.get_json()
        required_fields = ['firstName', 'lastName', 'email', 'password', 'role']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'message': f'{field} is required'}), 400

        if mongo_service.find_user_by_email(data['email']):
            return jsonify({'success': False, 'message': 'Email already exists'}), 409

        user_data = {
            'firstName': data['firstName'],
            'lastName': data['lastName'],
            'email': data['email'],
            'password': hash_password(data['password']),
            'phone': data.get('phone', ''),
            'location': data.get('location', ''),
            'farmSize': data.get('farmSize', ''),
            'role': data['role']
        }

        user_id = mongo_service.create_user(user_data)
        if user_id:
            log_action('create_user', f"Created user: {data['email']}", session.get('user_id'))
            return jsonify({'success': True, 'user_id': user_id})
        return jsonify({'success': False, 'message': 'Failed to create user'}), 500

    @app.route('/admin/api/users/<user_id>', methods=['PUT'])
    def admin_update_user_api(user_id):
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        data = request.get_json()
        allowed_fields = ['firstName', 'lastName', 'phone', 'location', 'farmSize', 'role']
        if 'password' in data and data['password']:
            data['password'] = hash_password(data['password'])
            allowed_fields.append('password')
        update_data = {k: v for k, v in data.items() if k in allowed_fields}

        if mongo_service.update_user(user_id, update_data):
            log_action('update_user', f"Updated user: {user_id}", session.get('user_id'))
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Failed to update user'}), 500

    @app.route('/admin/api/users/<user_id>', methods=['DELETE'])
    def admin_delete_user_api(user_id):
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        if mongo_service.delete_user(user_id):
            log_action('delete_user', f"Deleted user: {user_id}", session.get('user_id'))
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Failed to delete user'}), 500

    @app.route('/admin/api/diseases')
    def admin_diseases_api():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        if mongo_service is None:
            return jsonify({'success': False, 'error': 'Database not connected'}), 500

        diseases = mongo_service.find_diseases(limit=200)
        return jsonify({'success': True, 'diseases': diseases})

    @app.route('/admin/api/diseases', methods=['POST'])
    def admin_create_disease_api():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        data = request.get_json()
        if not data.get('name'):
            return jsonify({'success': False, 'message': 'Disease name is required'}), 400

        disease_id = mongo_service.insert_disease(data)
        if disease_id:
            log_action('create_disease', f"Created disease: {data['name']}", session.get('user_id'))
            return jsonify({'success': True, 'disease_id': disease_id})
        return jsonify({'success': False, 'message': 'Failed to create disease'}), 500

    @app.route('/admin/api/diseases/<disease_id>', methods=['PUT'])
    def admin_update_disease_api(disease_id):
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        data = request.get_json()
        if mongo_service.update_disease(disease_id, data):
            log_action('update_disease', f"Updated disease: {disease_id}", session.get('user_id'))
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Failed to update disease'}), 500

    @app.route('/admin/api/diseases/<disease_id>', methods=['DELETE'])
    def admin_delete_disease_api(disease_id):
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        if mongo_service.delete_disease(disease_id):
            log_action('delete_disease', f"Deleted disease: {disease_id}", session.get('user_id'))
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Failed to delete disease'}), 500

    @app.route('/admin/api/scans')
    def admin_scans_api():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        if mongo_service is None:
            return jsonify({'success': False, 'error': 'Database not connected'}), 500

        limit = int(request.args.get('limit', 50))
        skip = int(request.args.get('skip', 0))
        disease_filter = request.args.get('disease')

        scans = mongo_service.find_scans(limit=limit, skip=skip, disease_filter=disease_filter)
        total = mongo_service.count_scans(disease_filter=disease_filter)
        return jsonify({'success': True, 'scans': scans, 'total': total})

    @app.route('/admin/api/logs')
    def admin_logs_api():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'success': False, 'error': 'Access denied'}), 403

        if mongo_service is None:
            return jsonify({'success': False, 'error': 'Database not connected'}), 500

        limit = int(request.args.get('limit', 100))
        skip = int(request.args.get('skip', 0))
        action_filter = request.args.get('action')

        logs = mongo_service.find_logs(limit=limit, skip=skip, action_filter=action_filter)
        total = mongo_service.get_collection('logs').count_documents({}) if mongo_service.get_collection('logs') else 0
        return jsonify({'success': True, 'logs': logs, 'total': total})

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
