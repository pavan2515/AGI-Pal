import json
import os
from typing import List, Dict, Optional


class SchemesService:
    def __init__(self):
        self.schemes_dir = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'schemes'
        )

        self.scheme_files = {
            'agriculture': 'schemes_agriculture.json',
            'credit_finance': 'schemes_credit_finance.json',
            'energy_pump': 'schemes_energy_pump_support.json',
            'farm_infrastructure': 'schemes_farm_infrastructure.json',
            'fpo_capacity': 'schemes_fpo_capacity.json',
            'horticulture': 'schemes_horticulture_postharvest.json',
            'irrigation': 'schemes_irrigation_micro.json',
            'livestock': 'schemes_livestock_dairy_fisheries.json',
            'machinery': 'schemes_machinery_tools.json',
            'marketing': 'schemes_marketing_procurement.json',
            'organic': 'schemes_organic_sustainable.json',
            'processing': 'schemes_processing_value_add.json',
            'soil_health': 'schemes_soil_health_fertilizer.json',
            'storage': 'schemes_storage_coldchain.json',
            'women_shg': 'schemes_women_shg.json'
        }

        self._cache = {}

    # -----------------------------
    # INTERNAL FILE LOADER
    # -----------------------------
    def _load_scheme_file(self, category: str) -> List[Dict]:
        if category in self._cache:
            return self._cache[category]

        filename = self.scheme_files.get(category)
        if not filename:
            return []

        path = os.path.join(self.schemes_dir, filename)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                schemes = data.get('schemes', [])
                self._cache[category] = schemes
                return schemes
        except Exception as e:
            print(f"[ERROR] Loading {path}: {e}")
            return []

    # -----------------------------
    # RETURN ALL SCHEMES (FLATTENED)
    # -----------------------------
    def get_all_schemes(self) -> List[Dict]:
        all_schemes = []

        for category_key in self.scheme_files.keys():
            schemes = self._load_scheme_file(category_key)

            for scheme in schemes:
                scheme_copy = scheme.copy()
                scheme_copy['category_key'] = category_key
                scheme_copy['category_label'] = category_key.replace('_', ' ').title()
                all_schemes.append(scheme_copy)

        return all_schemes

    # -----------------------------
    # CATEGORY FILTER
    # -----------------------------
    def get_schemes_by_category(self, category: str) -> List[Dict]:
        schemes = self._load_scheme_file(category)
        result = []

        for scheme in schemes:
            scheme_copy = scheme.copy()
            scheme_copy['category_key'] = category
            scheme_copy['category_label'] = category.replace('_', ' ').title()
            result.append(scheme_copy)

        return result

    # -----------------------------
    # GET BY ID
    # -----------------------------
    def get_scheme_by_id(self, scheme_id: str) -> Optional[Dict]:
        for category in self.scheme_files.keys():
            schemes = self._load_scheme_file(category)
            for scheme in schemes:
                if scheme.get('scheme_id') == scheme_id:
                    scheme_copy = scheme.copy()
                    scheme_copy['category_key'] = category
                    scheme_copy['category_label'] = category.replace('_', ' ').title()
                    return scheme_copy
        return None

    # -----------------------------
    # SEARCH
    # -----------------------------
    def search_schemes(self, query: str) -> List[Dict]:
        query = query.lower()
        results = []

        for category in self.scheme_files.keys():
            schemes = self._load_scheme_file(category)
            for scheme in schemes:
                if (
                    query in scheme.get('name', '').lower()
                    or query in scheme.get('department', '').lower()
                    or query in scheme.get('benefit_summary', '').lower()
                ):
                    scheme_copy = scheme.copy()
                    scheme_copy['category_key'] = category
                    scheme_copy['category_label'] = category.replace('_', ' ').title()
                    results.append(scheme_copy)

        return results

    # -----------------------------
    # ADVANCED FILTER
    # -----------------------------
    def filter_schemes(
        self,
        category: Optional[str] = None,
        post_harvest: Optional[bool] = None,
        crop_type: Optional[str] = None
    ) -> List[Dict]:

        categories = [category] if category else self.scheme_files.keys()
        filtered = []

        for cat in categories:
            schemes = self._load_scheme_file(cat)

            for scheme in schemes:
                if post_harvest is not None:
                    if scheme.get('post_harvest_relevance') != post_harvest:
                        continue

                if crop_type:
                    crops = scheme.get('crop_type', [])
                    if 'All Crops' not in crops and crop_type not in crops:
                        continue

                scheme_copy = scheme.copy()
                scheme_copy['category_key'] = cat
                scheme_copy['category_label'] = cat.replace('_', ' ').title()
                filtered.append(scheme_copy)

        return filtered

    # -----------------------------
    # CATEGORY LIST (FOR BUTTONS)
    # -----------------------------
    def get_categories(self) -> List[Dict]:
        return [
            {
                'key': key,
                'name': key.replace('_', ' ').title()
            }
            for key in self.scheme_files.keys()
        ]
