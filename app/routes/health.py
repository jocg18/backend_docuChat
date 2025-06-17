from flask import Blueprint

bp = Blueprint('health', __name__)

@bp.route('/')
def index():
    return "Intelligent Document API is running!", 200

@bp.route('/health')
def health():
    return {"status": "ok"}, 200