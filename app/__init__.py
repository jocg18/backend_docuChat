from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    from app.routes.upload import bp as upload_bp
    from app.routes.query import bp as query_bp
    from app.routes.health import bp as health_bp
    from app.routes.image import bp as image_bp   #  ←  NUEVO
    from app.routes.enhanced_image import bp as enhanced_image_bp   #  ←  ENHANCED

    # Registra blueprints con los url_prefix deseados
    app.register_blueprint(upload_bp, url_prefix='/api')
    app.register_blueprint(query_bp, url_prefix='/api')
    app.register_blueprint(health_bp, url_prefix='')  # aquí sin prefijo para que / funcione
    app.register_blueprint(image_bp,   url_prefix="/api")  #  ←  NUEVO
    app.register_blueprint(enhanced_image_bp, url_prefix="/api/enhanced")  #  ←  ENHANCED

    return app
