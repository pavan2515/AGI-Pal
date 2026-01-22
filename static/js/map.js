// ==================== GLOBAL STATE ====================
let map;
let markers = [];
let farmerLat = null;
let farmerLon = null;
let currentPage = 'services';
let currentServiceType = null;

// ==================== LOCATION HANDLING ====================

/**
 * Get user's current location
 */
function getUserLocation() {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error('Geolocation is not supported by your browser'));
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {
                farmerLat = position.coords.latitude;
                farmerLon = position.coords.longitude;
                console.log(`✅ Location obtained: ${farmerLat}, ${farmerLon}`);
                resolve({ lat: farmerLat, lon: farmerLon });
            },
            (error) => {
                console.warn('⚠️ Geolocation error:', error.message);
                // Fallback to Mysuru, Karnataka coordinates
                farmerLat = 12.2958;
                farmerLon = 76.6394;
                resolve({ lat: farmerLat, lon: farmerLon });
            },
            {
                enableHighAccuracy: true,
                timeout: 8000,
                maximumAge: 0
            }
        );
    });
}

// ==================== SAFE DOM HELPERS ====================
function safeGet(id) {
    const el = document.getElementById(id);
    if (!el) console.warn(`⚠️ Element not found: #${id}`);
    return el;
}

function showElement(id) {
    const el = safeGet(id);
    if (el) el.style.display = 'block';
}

function hideElement(id) {
    const el = safeGet(id);
    if (el) el.style.display = 'none';
}

// ==================== NAVIGATION FUNCTIONS ====================

/**
 * Navigate to different services
 */
async function navigateToService(service) {
    hideElement('services-page');

    // Get location first
    try {
        await getUserLocation();
    } catch (error) {
        console.error('Location error:', error);
    }

    if (service === 'schemes') {
        loadSchemesPage();
    } else if (service === 'shops') {
        showShopTypeSelection();
    } else if (service === 'storage') {
        currentServiceType = 'storage';
        selectFacility('storage');
    } else if (service === 'markets') {
        currentServiceType = 'markets';
        selectFacility('market');
    } else {
        console.warn('⚠️ Unknown service:', service);
        backToServices();
    }
}

/**
 * Show shop type selection page
 */
function showShopTypeSelection() {
    currentPage = 'shop-selection';
    showElement('shop-type-page');
    hideElement('results-section');
    hideElement('schemes-page');
}

/**
 * Back to services page
 */
function backToServices() {
    currentPage = 'services';
    showElement('services-page');
    hideElement('shop-type-page');
    hideElement('results-section');
    hideElement('schemes-page');

    // Remove map to avoid duplicate init issues
    if (map) {
        map.remove();
        map = null;
    }
    markers = [];
}

/**
 * Go back based on current page
 */
function goBack() {
    if (currentPage === 'facilities') {
        if (currentServiceType === 'storage' || currentServiceType === 'markets') {
            backToServices();
        } else {
            showShopTypeSelection();
        }
    } else if (currentPage === 'shop-selection') {
        backToServices();
    } else if (currentPage === 'schemes') {
        backToServices();
    } else {
        backToServices();
    }
}

// ==================== FACILITIES & MAP FUNCTIONS ====================

/**
 * Called when user selects a facility type
 */
async function selectFacility(type) {
    currentPage = 'facilities';

    // Ensure we have location
    if (!farmerLat || !farmerLon) {
        await getUserLocation();
    }

    // Hide other pages
    hideElement('services-page');
    hideElement('shop-type-page');
    hideElement('schemes-page');
    showElement('results-section');

    // Update title
    updateResultsTitle(type);

    const listContent = safeGet("list-content");
    if (listContent) {
        listContent.innerHTML = `
            <div class="text-center py-12">
                <div class="loading-spinner mx-auto mb-4"></div>
                <p class="text-gray-600 font-medium">Loading nearby facilities...</p>
            </div>
        `;
    }

    // Destroy old map if exists
    if (map) {
        map.remove();
        map = null;
        markers = [];
    }

    // Leaflet must be loaded
    if (typeof L === "undefined") {
        console.error("❌ Leaflet is not loaded. Check Leaflet CDN script in HTML.");
        if (listContent) {
            listContent.innerHTML = `
                <div class="text-center py-12 px-4">
                    <i class="fas fa-exclamation-triangle text-red-500 text-5xl mb-4"></i>
                    <p class="text-red-600 font-semibold mb-2">Map library not loaded</p>
                    <p class="text-gray-600 text-sm">Leaflet CDN missing</p>
                </div>
            `;
        }
        return;
    }

    // Initialize map
    map = L.map("map").setView([farmerLat, farmerLon], 13);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "© OpenStreetMap contributors",
        maxZoom: 19
    }).addTo(map);

    // Farmer marker
    const farmerIcon = L.divIcon({
        className: 'custom-div-icon',
        html: '<div style="background: #16a34a; width: 30px; height: 30px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>',
        iconSize: [30, 30],
        iconAnchor: [15, 15]
    });

    L.marker([farmerLat, farmerLon], { icon: farmerIcon })
        .addTo(map)
        .bindPopup("<b>📍 Your Location</b>");

    // Decide backend endpoint
    let endpoint = "";
    if (type === "government" || type === "organic" || type === "chemical") {
        endpoint = "agro-shops";
    } else if (type === "market") {
        endpoint = "markets";
    } else if (type === "storage") {
        endpoint = "storage";
    } else {
        console.warn("⚠️ Unknown facility type:", type);
        return;
    }

    try {
        // ✅ IMPORTANT FIX: relative URL (works everywhere)
        const response = await fetch(`/post-harvest/${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                latitude: farmerLat,
                longitude: farmerLon,
                radius: 30
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        processResponse(data, type);

    } catch (error) {
        console.error('❌ Error fetching facilities:', error);
        if (listContent) {
            listContent.innerHTML = `
                <div class="text-center py-12 px-4">
                    <i class="fas fa-exclamation-triangle text-red-500 text-5xl mb-4"></i>
                    <p class="text-red-600 font-semibold mb-2">Failed to load data</p>
                    <p class="text-gray-600 text-sm mb-4">Please check your backend connection</p>
                    <button onclick="selectFacility('${type}')" class="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
                        <i class="fas fa-redo mr-2"></i>Retry
                    </button>
                </div>
            `;
        }
    }
}

/**
 * Update results page title based on service type
 */
function updateResultsTitle(type) {
    const titleMap = {
        'government': '🏛️ Government Agro Shops',
        'chemical': '🧪 Private Chemical Shops',
        'organic': '🌿 Organic Input Shops',
        'market': '🏪 Nearby Markets',
        'storage': '❄️ Cold Storage Facilities'
    };

    const subtitleMap = {
        'government': 'Subsidized inputs and government support',
        'chemical': 'Pesticides, fertilizers, and agricultural chemicals',
        'organic': 'Natural and organic farming solutions',
        'market': 'Best prices for your agricultural produce',
        'storage': 'Preserve your harvest with cold storage'
    };

    const titleEl = safeGet('results-title');
    const subtitleEl = safeGet('results-subtitle');

    if (titleEl) titleEl.innerHTML = titleMap[type] || 'Nearby Facilities';
    if (subtitleEl) subtitleEl.innerHTML = subtitleMap[type] || 'Click on any facility to view details';
}

/**
 * Process backend response and filter
 */
function processResponse(data, type) {
    let items = [];

    if (type === "government") {
        items = data.government_shops || [];
    } else if (type === "organic") {
        items = data.organic_shops || [];
    } else if (type === "chemical") {
        items = [
            ...(data.government_shops || []),
            ...(data.organic_shops || [])
        ].filter(shop =>
            shop.services &&
            shop.services.some(s =>
                String(s).toLowerCase().includes("pesticide") ||
                String(s).toLowerCase().includes("fertilizer") ||
                String(s).toLowerCase().includes("chemical")
            )
        );
    } else if (type === "market") {
        items = data.markets || [];
    } else if (type === "storage") {
        items = data.cold_storage || [];
    }

    console.log(`✅ Found ${items.length} ${type} facilities`);
    renderListAndMarkers(items, type);
}

/**
 * Render list and markers on map
 */
function renderListAndMarkers(items, type) {
    const listContainer = safeGet("list-content");
    if (!listContainer) return;

    listContainer.innerHTML = "";

    // Clear existing markers
    if (map) {
        markers.forEach(m => map.removeLayer(m));
    }
    markers = [];

    if (!items || items.length === 0) {
        listContainer.innerHTML = `
            <div class="text-center py-12">
                <i class="fas fa-search text-gray-400 text-5xl mb-4"></i>
                <p class="text-gray-600 font-semibold mb-2">No facilities found nearby</p>
                <p class="text-gray-500 text-sm">Try expanding your search radius or check another location</p>
            </div>
        `;
        return;
    }

    // Add header with count
    const header = document.createElement('div');
    header.className = 'bg-green-50 p-4 rounded-lg mb-4 border border-green-200';
    header.innerHTML = `
        <p class="text-green-800 font-semibold">
            <i class="fas fa-map-marker-alt mr-2"></i>Found ${items.length} facilities within 30 km
        </p>
    `;
    listContainer.appendChild(header);

    const getMarkerIcon = (t) => {
        const iconMap = {
            'government': '🏛️',
            'organic': '🌿',
            'chemical': '🧪',
            'market': '🏪',
            'storage': '❄️'
        };
        return iconMap[t] || '📍';
    };

    items.forEach((item) => {
        if (!item.latitude || !item.longitude) return;

        // Create custom marker icon
        const markerIcon = L.divIcon({
            className: 'custom-div-icon',
            html: `<div style="font-size: 24px;">${getMarkerIcon(type)}</div>`,
            iconSize: [30, 30],
            iconAnchor: [15, 30]
        });

        // Marker
        const marker = L.marker([item.latitude, item.longitude], { icon: markerIcon }).addTo(map);

        const popupContent = `
            <div class="p-2" style="min-width: 200px;">
                <h3 class="font-bold text-base mb-2">${item.name || "Facility"}</h3>
                ${item.address ? `<p class="text-sm text-gray-600 mb-1">📍 ${item.address}</p>` : ''}
                ${item.phone ? `<p class="text-sm text-gray-600 mb-2">📞 ${item.phone}</p>` : ''}
                ${item.distance_km ? `
                    <div class="bg-green-100 text-green-800 px-2 py-1 rounded text-sm font-semibold inline-block">
                        📏 ${item.distance_km} km away
                    </div>` : ''
                }
            </div>
        `;

        marker.bindPopup(popupContent);
        markers.push(marker);

        // List item
        const div = document.createElement("div");
        div.className =
            "list-item bg-white p-5 mb-4 rounded-xl shadow-md cursor-pointer hover:shadow-xl transition-all border-l-4 border-green-500";

        let servicesBadges = '';
        if (item.services && item.services.length > 0) {
            servicesBadges = `
                <div class="flex flex-wrap gap-2 mb-3">
                    ${item.services.slice(0, 3).map(s => `
                        <span class="badge text-xs bg-blue-50 text-blue-700 px-3 py-1 rounded-full font-medium">
                            <i class="fas fa-check-circle"></i> ${s}
                        </span>
                    `).join('')}
                    ${item.services.length > 3 ? `<span class="text-xs text-gray-500">+${item.services.length - 3} more</span>` : ''}
                </div>
            `;
        }

        const safeName = String(item.name || "Facility").replace(/'/g, "\\'");

        div.innerHTML = `
            <div class="flex justify-between items-start mb-3">
                <div class="flex-1">
                    <h3 class="font-bold text-lg text-gray-800 mb-1">${item.name || "Facility"}</h3>
                    ${item.distance_km ? `
                        <div class="inline-block distance-badge text-white text-xs px-3 py-1 rounded-full font-semibold">
                            <i class="fas fa-location-arrow mr-1"></i>${item.distance_km} km away
                        </div>` : ''
                    }
                </div>
                <div class="text-2xl">${getMarkerIcon(type)}</div>
            </div>

            ${item.address ? `
                <p class="text-sm text-gray-600 mb-2 flex items-start gap-2">
                    <i class="fas fa-map-marker-alt text-red-500 mt-1"></i>
                    <span>${item.address}</span>
                </p>` : ''
            }

            ${item.phone ? `
                <p class="text-sm text-gray-600 mb-3 flex items-center gap-2">
                    <i class="fas fa-phone text-green-600"></i>
                    <a href="tel:${item.phone}" class="hover:text-green-600">${item.phone}</a>
                </p>` : ''
            }

            ${servicesBadges}

            ${item.capacity ? `
                <p class="text-sm text-gray-600 mb-3 flex items-center gap-2">
                    <i class="fas fa-warehouse text-blue-600"></i>
                    <span>Capacity: ${item.capacity}</span>
                </p>` : ''
            }

            <button onclick="event.stopPropagation(); navigate(${item.latitude}, ${item.longitude}, '${safeName}')" 
                    class="navigate-btn w-full px-4 py-3 text-white rounded-lg font-semibold flex items-center justify-center gap-2 mt-3">
                <i class="fas fa-directions"></i>
                Get Directions
            </button>
        `;

        // Sync list click → map
        div.onclick = (e) => {
            if (!e.target.closest('button') && !e.target.closest('a')) {
                marker.openPopup();
                map.setView([item.latitude, item.longitude], 15);
            }
        };

        listContainer.appendChild(div);
    });

    // Fit map
    if (markers.length > 0) {
        const group = new L.featureGroup(markers);
        map.fitBounds(group.getBounds().pad(0.1));
    }
}

/**
 * Navigate using Google Maps
 */
function navigate(lat, lon, name) {
    const originLat = farmerLat ?? 12.2958;
    const originLon = farmerLon ?? 76.6394;
    const url = `https://www.google.com/maps/dir/?api=1&origin=${originLat},${originLon}&destination=${lat},${lon}&destination_place_id=${encodeURIComponent(name)}`;
    window.open(url, "_blank");
}

// ==================== SCHEMES FUNCTIONALITY ====================

async function loadSchemesPage() {
    currentPage = 'schemes';

    hideElement('services-page');
    hideElement('shop-type-page');
    hideElement('results-section');

    const schemesPage = safeGet('schemes-page');
    if (!schemesPage) return;
    schemesPage.style.display = 'block';

    schemesPage.innerHTML = `
        <div class="min-h-screen bg-gray-50">
            <div class="bg-gradient-to-r from-orange-600 to-orange-700 text-white py-12">
                <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <button onclick="backToServices()" class="flex items-center text-white mb-6 px-4 py-2 rounded-lg back-btn hover:bg-orange-600 transition-colors">
                        <i class="fas fa-arrow-left mr-2"></i>
                        Back to Services
                    </button>
                    <h1 class="text-4xl font-bold mb-2">
                        <i class="fas fa-file-contract mr-3"></i>Government Schemes
                    </h1>
                    <p class="text-orange-100 text-lg">Explore and apply for various agricultural schemes and subsidies</p>
                </div>
            </div>

            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div class="relative mb-8">
                    <input 
                        type="text" 
                        id="scheme-search" 
                        placeholder="Search schemes by name, department, or keywords..." 
                        class="w-full px-4 py-3 pl-12 border-2 border-gray-300 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-orange-500 shadow-sm"
                        onkeyup="searchSchemes()"
                    />
                    <i class="fas fa-search absolute left-4 top-4 text-gray-400"></i>
                </div>

                <div class="flex gap-3 flex-wrap mb-8" id="category-filters">
                    <button onclick="filterByCategory('all')" class="px-5 py-2.5 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors font-semibold shadow-md">
                        <i class="fas fa-th mr-2"></i>All Schemes
                    </button>
                </div>

                <div id="schemes-loading" class="text-center py-16 hidden">
                    <div class="loading-spinner mx-auto mb-4"></div>
                    <p class="text-gray-600 font-medium">Loading schemes...</p>
                </div>

                <div id="schemes-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"></div>
            </div>
        </div>

        <div id="scheme-modal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4" onclick="closeSchemeModalOnBackdrop(event)">
            <div class="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl" onclick="event.stopPropagation()">
                <div id="modal-content"></div>
            </div>
        </div>
    `;

    await loadSchemes();
    await loadCategories();
}

async function loadSchemes() {
    const loadingDiv = safeGet('schemes-loading');
    const gridDiv = safeGet('schemes-grid');

    if (loadingDiv) loadingDiv.classList.remove('hidden');
    if (gridDiv) gridDiv.innerHTML = '';

    try {
        // ✅ FIX: relative URL
        const response = await fetch('/api/schemes');
        const data = await response.json();

        if (data.success) {
            let allSchemes = [];

            // ✅ support both formats: grouped object OR flat list
            if (Array.isArray(data.schemes)) {
                allSchemes = data.schemes;
            } else if (typeof data.schemes === "object") {
                Object.entries(data.schemes).forEach(([category, schemes]) => {
                    (schemes || []).forEach(scheme => {
                        allSchemes.push({ ...scheme, category });
                    });
                });
            }

            displaySchemes(allSchemes);
        } else {
            throw new Error("API returned success=false");
        }

    } catch (error) {
        console.error('Error loading schemes:', error);
        if (gridDiv) {
            gridDiv.innerHTML = `
                <div class="col-span-full text-center py-12">
                    <i class="fas fa-exclamation-triangle text-red-500 text-5xl mb-4"></i>
                    <p class="text-red-600 font-semibold mb-2">Error loading schemes</p>
                    <p class="text-gray-600 text-sm">Please check your backend connection</p>
                </div>
            `;
        }
    } finally {
        if (loadingDiv) loadingDiv.classList.add('hidden');
    }
}

async function loadCategories() {
    try {
        // ✅ FIX: relative URL
        const response = await fetch('/api/schemes/categories');
        const data = await response.json();

        if (data.success) {
            const filtersContainer = safeGet('category-filters');
            if (!filtersContainer) return;

            data.categories.forEach(cat => {
                const button = document.createElement('button');
                button.innerHTML = `<i class="fas fa-tag mr-2"></i>${cat.name}`;
                button.className = 'px-5 py-2.5 bg-white border-2 border-gray-300 rounded-lg hover:bg-orange-50 hover:border-orange-500 transition-colors font-semibold shadow-sm';
                button.onclick = () => filterByCategory(cat.key);
                filtersContainer.appendChild(button);
            });
        }
    } catch (error) {
        console.error('Error loading categories:', error);
    }
}

function displaySchemes(schemes) {
    const grid = safeGet('schemes-grid');
    if (!grid) return;

    if (!schemes || schemes.length === 0) {
        grid.innerHTML = `
            <div class="col-span-full text-center py-12">
                <i class="fas fa-inbox text-gray-400 text-5xl mb-4"></i>
                <p class="text-gray-600 font-semibold">No schemes found</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = schemes.map(scheme => {
        const schemeJson = JSON.stringify(scheme).replace(/"/g, '&quot;');
        return `
            <div class="bg-white rounded-xl shadow-md hover:shadow-2xl transition-all p-6 cursor-pointer border-2 border-transparent hover:border-orange-500"
                 onclick='showSchemeDetail(${schemeJson})'>
                <div class="flex justify-between items-start mb-3">
                    <h3 class="text-xl font-bold text-gray-800 flex-1 pr-3">${scheme.name || 'Scheme'}</h3>
                    <i class="fas fa-arrow-right text-orange-600 text-xl"></i>
                </div>

                ${scheme.category_label ? `
                    <span class="inline-block bg-orange-100 text-orange-800 text-xs px-3 py-1 rounded-full font-semibold mb-3">
                        <i class="fas fa-tag mr-1"></i>${scheme.category_label}
                    </span>
                ` : ''}

                <p class="text-gray-600 mb-4 text-sm line-clamp-3">${scheme.description || 'Click to view details'}</p>

                ${scheme.benefits ? `
                    <div class="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-3 mb-4 border-l-4 border-blue-500">
                        <p class="text-sm text-blue-900 font-medium">
                            <i class="fas fa-gift mr-2"></i>${scheme.benefits.substring(0, 100)}${scheme.benefits.length > 100 ? '...' : ''}
                        </p>
                    </div>
                ` : ''}

                <div class="flex items-center justify-between text-sm text-gray-500 pt-4 border-t">
                    <span><i class="fas fa-phone mr-2"></i>${scheme.contact_number || 'Contact for details'}</span>
                    ${scheme.website ? '<i class="fas fa-external-link-alt text-orange-600"></i>' : ''}
                </div>
            </div>
        `;
    }).join('');
}

// ---------- MODAL ----------
function showSchemeDetail(scheme) {
    const modal = safeGet('scheme-modal');
    const content = safeGet('modal-content');
    if (!modal || !content) return;

    content.innerHTML = `
        <div class="sticky top-0 bg-gradient-to-r from-orange-600 to-orange-700 text-white p-6 flex justify-between items-start rounded-t-2xl">
            <div class="flex-1 pr-4">
                <h2 class="text-3xl font-bold mb-2">${scheme.name || 'Scheme'}</h2>
                ${scheme.category_label ? `
                    <span class="inline-block bg-white bg-opacity-20 text-white text-sm px-4 py-1.5 rounded-full font-semibold">
                        <i class="fas fa-tag mr-1"></i>${scheme.category_label}
                    </span>
                ` : ''}
            </div>
            <button onclick="closeSchemeModal()" class="text-white hover:bg-white hover:bg-opacity-20 p-2 rounded-full transition-colors">
                <i class="fas fa-times text-2xl"></i>
            </button>
        </div>

        <div class="p-8 space-y-6">
            ${scheme.description ? `
                <div class="bg-gray-50 rounded-xl p-5 border border-gray-200">
                    <h3 class="text-xl font-bold text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-info-circle text-blue-600 mr-3"></i>Description
                    </h3>
                    <p class="text-gray-700 leading-relaxed">${scheme.description}</p>
                </div>
            ` : ''}

            ${scheme.eligibility ? `
                <div class="bg-yellow-50 rounded-xl p-5 border border-yellow-200">
                    <h3 class="text-xl font-bold text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-user-check text-yellow-600 mr-3"></i>Eligibility Criteria
                    </h3>
                    <p class="text-gray-700 leading-relaxed">${scheme.eligibility}</p>
                </div>
            ` : ''}

            ${scheme.benefits ? `
                <div class="bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl p-5 border-l-4 border-blue-600">
                    <h3 class="text-xl font-bold text-blue-900 mb-3 flex items-center">
                        <i class="fas fa-gift text-blue-700 mr-3"></i>Benefits
                    </h3>
                    <p class="text-blue-900 leading-relaxed font-medium">${scheme.benefits}</p>
                </div>
            ` : ''}

            ${scheme.how_to_apply ? `
                <div class="bg-green-50 rounded-xl p-5 border border-green-200">
                    <h3 class="text-xl font-bold text-gray-800 mb-3 flex items-center">
                        <i class="fas fa-clipboard-list text-green-600 mr-3"></i>How to Apply
                    </h3>
                    <p class="text-gray-700 leading-relaxed">${scheme.how_to_apply}</p>
                </div>
            ` : ''}

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                ${scheme.contact_number ? `
                    <div class="flex items-center bg-green-50 rounded-xl p-5 border border-green-200 hover:shadow-md transition-shadow">
                        <div class="bg-green-600 text-white p-3 rounded-full mr-4">
                            <i class="fas fa-phone text-2xl"></i>
                        </div>
                        <div>
                            <p class="text-sm font-semibold text-gray-700">Helpline Number</p>
                            <a href="tel:${scheme.contact_number}" class="text-green-700 font-bold text-lg hover:text-green-800">${scheme.contact_number}</a>
                        </div>
                    </div>
                ` : ''}

                ${scheme.website ? `
                    <a href="${scheme.website}" target="_blank" class="flex items-center bg-blue-50 rounded-xl p-5 border border-blue-200 hover:shadow-md hover:bg-blue-100 transition-all">
                        <div class="bg-blue-600 text-white p-3 rounded-full mr-4">
                            <i class="fas fa-globe text-2xl"></i>
                        </div>
                        <div>
                            <p class="text-sm font-semibold text-gray-700">Official Website</p>
                            <p class="text-blue-700 font-bold text-lg">Visit & Apply Online →</p>
                        </div>
                    </a>
                ` : ''}
            </div>
        </div>

        <div class="sticky bottom-0 bg-gray-50 border-t-2 p-6 flex justify-end gap-3 rounded-b-2xl">
            ${scheme.website ? `
                <a href="${scheme.website}" target="_blank" class="px-8 py-3 bg-gradient-to-r from-orange-600 to-orange-700 text-white rounded-lg hover:from-orange-700 hover:to-orange-800 font-semibold shadow-md transition-all flex items-center gap-2">
                    <i class="fas fa-external-link-alt"></i>
                    Visit Official Website
                </a>
            ` : ''}
            <button onclick="closeSchemeModal()" class="px-8 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-semibold shadow-md transition-all">
                <i class="fas fa-times mr-2"></i>Close
            </button>
        </div>
    `;

    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeSchemeModal() {
    const modal = safeGet('scheme-modal');
    if (modal) {
        modal.classList.add('hidden');
        document.body.style.overflow = 'auto';
    }
}

function closeSchemeModalOnBackdrop(event) {
    if (event.target.id === 'scheme-modal') {
        closeSchemeModal();
    }
}

// ---------- SEARCH & FILTER ----------
async function searchSchemes() {
    const searchInput = safeGet('scheme-search');
    if (!searchInput) return;

    const query = searchInput.value;
    if (query.length < 2) {
        await loadSchemes();
        return;
    }

    const gridDiv = safeGet('schemes-grid');
    if (gridDiv) {
        gridDiv.innerHTML = '<div class="col-span-full text-center py-12"><div class="loading-spinner mx-auto"></div></div>';
    }

    try {
        const response = await fetch(`/api/schemes/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();

        if (data.success) {
            displaySchemes(data.results || []);
        }
    } catch (error) {
        console.error('Error searching schemes:', error);
        if (gridDiv) {
            gridDiv.innerHTML = '<div class="col-span-full text-center py-12"><p class="text-red-600">Search failed. Please try again.</p></div>';
        }
    }
}

async function filterByCategory(category) {
    const gridDiv = safeGet('schemes-grid');
    if (gridDiv) {
        gridDiv.innerHTML = '<div class="col-span-full text-center py-12"><div class="loading-spinner mx-auto"></div></div>';
    }

    try {
        if (category === 'all') {
            await loadSchemes();
        } else {
            const response = await fetch(`/api/schemes/category/${category}`);
            const data = await response.json();

            if (data.success) {
                displaySchemes(data.schemes || []);
            }
        }
    } catch (error) {
        console.error('Error filtering schemes:', error);
        if (gridDiv) {
            gridDiv.innerHTML = '<div class="col-span-full text-center py-12"><p class="text-red-600">Filter failed. Please try again.</p></div>';
        }
    }
}

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Post-Harvest Management System Loaded');

    getUserLocation()
        .then(location => console.log('✅ Initial location set:', location))
        .catch(error => console.error('❌ Failed to get initial location:', error));
});
