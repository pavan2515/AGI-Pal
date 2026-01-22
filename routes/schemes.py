from flask import Blueprint, request, jsonify
from services.schemes_service import SchemesService

schemes_bp = Blueprint('schemes', __name__, url_prefix='/api')
schemes_service = SchemesService()


@schemes_bp.route('/schemes', methods=['GET'])
def get_schemes():
    """
    Get all schemes OR filtered schemes.
    Frontend expects grouped-by-category structure.
    """
    try:
        category = request.args.get('category')
        post_harvest = request.args.get('post_harvest')
        crop_type = request.args.get('crop_type')

        # Convert post_harvest to boolean if provided
        if post_harvest is not None:
            post_harvest = post_harvest.lower() == 'true'

        # Apply filters if any parameter is provided
        if category or post_harvest is not None or crop_type:
            schemes = schemes_service.filter_schemes(
                category=category,
                post_harvest=post_harvest,
                crop_type=crop_type
            )

            return jsonify({
                'success': True,
                'count': len(schemes),
                'schemes': schemes
            })

        # Default: return all schemes grouped by category
        all_schemes = schemes_service.get_all_schemes()

        total_count = sum(len(v) for v in all_schemes.values())

        return jsonify({
            'success': True,
            'total_count': total_count,
            'schemes': all_schemes
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@schemes_bp.route('/schemes/categories', methods=['GET'])
def get_categories():
    """
    Return categories exactly as frontend expects:
    [{ key, name }]
    """
    try:
        categories = schemes_service.get_categories()

        return jsonify({
            'success': True,
            'count': len(categories),
            'categories': categories
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@schemes_bp.route('/schemes/category/<category>', methods=['GET'])
def get_schemes_by_category(category):
    """
    Get schemes for one category key
    """
    try:
        schemes = schemes_service.get_schemes_by_category(category)

        if not schemes:
            return jsonify({
                'success': False,
                'message': f'No schemes found for category: {category}'
            }), 404

        return jsonify({
            'success': True,
            'category': category,
            'count': len(schemes),
            'schemes': schemes
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@schemes_bp.route('/schemes/<scheme_id>', methods=['GET'])
def get_scheme_by_id(scheme_id):
    """
    Fetch scheme by ID (search across all JSON files)
    """
    try:
        scheme = schemes_service.get_scheme_by_id(scheme_id)

        if not scheme:
            return jsonify({
                'success': False,
                'message': f'Scheme not found: {scheme_id}'
            }), 404

        return jsonify({
            'success': True,
            'scheme': scheme
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@schemes_bp.route('/schemes/search', methods=['GET'])
def search_schemes():
    """
    Keyword-based search across all schemes
    """
    try:
        query = request.args.get('q', '').strip()

        if not query:
            return jsonify({
                'success': False,
                'message': 'Search query (q) is required'
            }), 400

        if len(query) < 2:
            return jsonify({
                'success': False,
                'message': 'Search query must be at least 2 characters'
            }), 400

        results = schemes_service.search_schemes(query)

        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
