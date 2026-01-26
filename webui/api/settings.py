"""
Settings Schema API

Provides endpoints to access the centralized settings schema,
enabling dynamic form generation and validation in the WebUI.
"""

from flask import Blueprint, jsonify
from ..settings_schema import (
    get_settings_schema,
    get_settings_by_category,
    get_setting_by_key,
    get_defaults_for_category,
    SettingCategory
)

settings_bp = Blueprint('settings', __name__, url_prefix='/api/settings')


@settings_bp.route('/schema', methods=['GET'])
def get_schema():
    """
    Get complete settings schema.
    
    Returns the full schema with all categories and settings definitions.
    Used for initial page load and form generation.
    
    Response:
        {
            "version": "1.0",
            "categories": {
                "audio": {
                    "label": "Audio",
                    "settings": ["force_mono", "silence_threshold", ...]
                },
                ...
            },
            "settings": {
                "force_mono": {
                    "key": "force_mono",
                    "type": "bool",
                    "default": true,
                    "label": "Force Mono",
                    "description": "...",
                    ...
                },
                ...
            }
        }
    """
    try:
        schema = get_settings_schema()
        return jsonify(schema), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to load settings schema',
            'message': str(e)
        }), 500


@settings_bp.route('/category/<category_name>', methods=['GET'])
def get_category_settings(category_name: str):
    """
    Get settings for a specific category.
    
    Args:
        category_name: Category identifier (e.g., 'midi_output', 'kick')
    
    Returns:
        List of setting definitions in that category
    """
    try:
        # Validate category
        try:
            category = SettingCategory(category_name)
        except ValueError:
            return jsonify({
                'error': 'Invalid category',
                'message': f'Unknown category: {category_name}',
                'valid_categories': [c.value for c in SettingCategory]
            }), 400
        
        settings = get_settings_by_category(category)
        
        return jsonify({
            'category': category_name,
            'settings': [s.to_dict() for s in settings]
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to load category settings',
            'message': str(e)
        }), 500


@settings_bp.route('/defaults/<category_name>', methods=['GET'])
def get_category_defaults(category_name: str):
    """
    Get default values for a category.
    
    Args:
        category_name: Category identifier
    
    Returns:
        Dictionary of setting keys to default values
    """
    try:
        # Validate category
        try:
            category = SettingCategory(category_name)
        except ValueError:
            return jsonify({
                'error': 'Invalid category',
                'message': f'Unknown category: {category_name}'
            }), 400
        
        defaults = get_defaults_for_category(category)
        
        return jsonify({
            'category': category_name,
            'defaults': defaults
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to load defaults',
            'message': str(e)
        }), 500


@settings_bp.route('/setting/<setting_key>', methods=['GET'])
def get_setting(setting_key: str):
    """
    Get definition for a specific setting.
    
    Args:
        setting_key: Setting identifier
    
    Returns:
        Setting definition
    """
    try:
        setting = get_setting_by_key(setting_key)
        
        if not setting:
            return jsonify({
                'error': 'Setting not found',
                'message': f'No setting with key: {setting_key}'
            }), 404
        
        return jsonify(setting.to_dict()), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Failed to load setting',
            'message': str(e)
        }), 500


@settings_bp.route('/validate', methods=['POST'])
def validate_settings():
    """
    Validate settings values.
    
    Request body:
        {
            "settings": {
                "onset_threshold": 0.3,
                "min_velocity": 80,
                ...
            }
        }
    
    Response:
        {
            "valid": true,
            "errors": {}
        }
        
        or
        
        {
            "valid": false,
            "errors": {
                "onset_threshold": "Value must be >= 0",
                "min_velocity": "Value must be <= 127"
            }
        }
    """
    from flask import request
    
    try:
        data = request.get_json()
        
        if not data or 'settings' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must include "settings" object'
            }), 400
        
        settings_to_validate = data['settings']
        errors = {}
        
        for key, value in settings_to_validate.items():
            setting_def = get_setting_by_key(key)
            
            if not setting_def:
                errors[key] = f'Unknown setting: {key}'
                continue
            
            is_valid, error_msg = setting_def.validate(value)
            if not is_valid:
                errors[key] = error_msg
        
        return jsonify({
            'valid': len(errors) == 0,
            'errors': errors
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Validation failed',
            'message': str(e)
        }), 500
