function getBaseUrl() {
  const baseTag = document.querySelector('base');
  if (baseTag && baseTag.href) {
    return new URL(baseTag.href).pathname;
  }
  const pathParts = window.location.pathname.split('/');
  if (pathParts.length > 2 && pathParts[1] === 'holohub') {
    return '/holohub/';
  }
  return '';
}

// Add CSS to hide initial non-dynamic content until ready
(function addInitialStyles() {
  const style = document.createElement('style');
  style.id = 'tag-sidebar-initial-styles';
  style.textContent = `
    .md-content__inner {
      opacity: 0;
      transition: opacity 0.3s ease-in-out;
    }

    body.content-ready .md-content__inner {
      opacity: 1;
    }

    .tag-sidebar-loading-indicator {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      z-index: 999;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 5px;
      padding: 20px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      text-align: center;
      transition: opacity 0.3s ease-out;
    }

    .tag-sidebar-loading-indicator.hidden {
      opacity: 0;
      pointer-events: none;
    }

    .tag-sidebar-loading-spinner {
      display: inline-block;
      width: 30px;
      height: 30px;
      border: 3px solid rgba(0, 0, 0, 0.1);
      border-radius: 50%;
      border-top-color: #2196F3;
      animation: spin 1s ease-in-out infinite;
      margin-bottom: 10px;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Prepare sidebar structure early */
    .tag-sidebar {
      opacity: 0;
      transition: opacity 0.3s ease-in-out;
    }

    body.tag-sidebar-ready .tag-sidebar {
      opacity: 1;
    }
  `;

  document.head.appendChild(style);
})();

// Create loading indicator
function createLoadingIndicator() {
  const loadingEl = document.createElement('div');
  loadingEl.className = 'tag-sidebar-loading-indicator';
  loadingEl.innerHTML = `
    <div class="tag-sidebar-loading-spinner"></div>
    <div>Loading application data...</div>
  `;
  document.body.appendChild(loadingEl);

  return {
    el: loadingEl,
    hide: function() {
      loadingEl.classList.add('hidden');
      setTimeout(() => {
        loadingEl.remove();
      }, 300);
    }
  };
}

let loading = null;
let preloadStarted = false;

function isTagsPage() {
  return window.location.pathname.endsWith('/tags/') ||
         window.location.pathname.endsWith('/tags') ||
         window.location.pathname.includes('/tags/index');
}

// Function to check if we're on a page that needs the sidebar
function needsSidebar() {
  const isTagsPageResult = isTagsPage();
  const isHoloHub = window.location.pathname.endsWith('/holohub/');
  return isHoloHub || isTagsPageResult;
}

// Ensure content is visible regardless of sidebar status
function makeContentVisible() {
  document.body.classList.add('content-ready');
  if (loading) {
    loading.hide();
  }
}

// Unified function to initialize sidebar if needed
function initializeIfNeeded() {
  if (preloadStarted) {
    return; // Already initialized or in progress
  }
  if (needsSidebar()) {
    preloadStarted = true;
    loading = createLoadingIndicator();
    preloadData();
  } else {
    makeContentVisible();
  }
}

document.addEventListener('readystatechange', function() {
  if (document.readyState === 'interactive') {
    initializeIfNeeded();
  }
}, { once: true });

// Begin preload immediately if document is already interactive or complete
if (document.readyState === 'interactive' || document.readyState === 'complete') {
  initializeIfNeeded();
}

// Function to fetch data with proper caching parameters
async function fetchDataWithCache(url, forceRefresh = false) {
  const cacheParam = forceRefresh ? `?v=${Date.now()}` : '';
  try {
    const response = await fetch(`${url}${cacheParam}`, {
      headers: {
        'Cache-Control': forceRefresh ? 'no-cache' : '',
        'Pragma': forceRefresh ? 'no-cache' : ''
      }
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch data: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching ${url}:`, error);
    return null;
  }
}

function getUrlParams() {
  return new URLSearchParams(window.location.search);
}

function isForceRefreshRequested() {
  return getUrlParams().has('refresh_cache');
}

// Allow triggering a reload via URL
function checkForCacheRefreshParam() {
  if (isForceRefreshRequested() && needsSidebar()) {
    const urlParams = getUrlParams();
    urlParams.delete('refresh_cache');
    const newUrl = window.location.pathname + (urlParams.toString() ? '?' + urlParams.toString() : '');
    window.history.replaceState({}, document.title, newUrl);

    return window.refreshTagSidebarCache();
  }
  return Promise.resolve(false);
}

window.refreshTagSidebarCache = async function() {
  console.log("Force refreshing tag sidebar cache");
  const baseUrl = getBaseUrl();
  let dataPath = `${baseUrl}_data/`;

  dataLoadPromise = null;
  window.tagSidebarData = {
    categories: null,
    appCardsData: null,
    isLoading: true,
    preloadStarted: false,
    preloadComplete: false
  };

  try {
    const data = await loadDataWithCacheShared(dataPath, true);
    window.tagSidebarData.categories = data.categories;
    window.tagSidebarData.appCardsData = data.appCardsData;
    window.tagSidebarData.preloadComplete = true;
    console.log("Cache refresh complete");
    return true;
  } catch (error) {
    console.error("Cache refresh failed:", error);
    return false;
  } finally {
    window.tagSidebarData.isLoading = false;
  }
};

// Immediately start preloading data as early as possible
function preloadData() {
  if (!needsSidebar()) {
    makeContentVisible();
    return;
  }

  console.log("Preloading tag sidebar data...");
  window.tagSidebarData = window.tagSidebarData || {
    categories: null,
    appCardsData: null,
    isLoading: false,
    preloadStarted: false,
    preloadComplete: false
  };
  if (window.tagSidebarData.preloadStarted) {
    return;
  }

  window.tagSidebarData.preloadStarted = true;
  window.tagSidebarData.isLoading = true;

  // Get the base URL early
  const baseUrl = getBaseUrl();
  const dataPath = `${baseUrl}_data/`;
  const forceRefresh = isForceRefreshRequested();

  window.tagSidebarData.categoriesPromise = fetchDataWithCache(`${dataPath}tmp_tag-categories.json`, forceRefresh);
  window.tagSidebarData.appCardsPromise = fetchDataWithCache(`${dataPath}tmp_app_cards.json`, forceRefresh);

  // Process both promises together to resolve data
  Promise.allSettled([
    window.tagSidebarData.categoriesPromise,
    window.tagSidebarData.appCardsPromise
  ]).then(results => {
    if (results[0].status === 'fulfilled' && results[0].value) {
      window.tagSidebarData.categories = results[0].value;
      console.log("Categories preloaded:", window.tagSidebarData.categories.length);
    }

    if (results[1].status === 'fulfilled' && results[1].value) {
      window.tagSidebarData.appCardsData = results[1].value;
      console.log("App cards preloaded:", Object.keys(window.tagSidebarData.appCardsData || {}).length);
    }

    window.tagSidebarData.isLoading = false;
    window.tagSidebarData.preloadComplete = true;

    if (document.readyState === 'interactive' || document.readyState === 'complete') {
      initializeUI();
    }
  });
}

// Initialize UI when data is available
async function initializeUI() {
  if (!needsSidebar()) {
    makeContentVisible(); // Ensure content is visible
    return;
  }
  try {
    await checkForCacheRefreshParam();
    const baseUrl = getBaseUrl();

    if (!globalTagPopup) {
      globalTagPopup = new TagPopup();
    }
    window.showAllTags = function(element, allTags) {
      const tags = JSON.parse(allTags);
      if (globalTagPopup) {
        globalTagPopup.show(element, tags);
      } else {
        console.error("Tag popup not initialized");
      }
      return false;
    };
    if (window.tagSidebarData.isLoading) {
      console.log("Waiting for data to complete loading...");
      await new Promise(resolve => {
        const checkInterval = setInterval(() => {
          if (!window.tagSidebarData.isLoading) {
            clearInterval(checkInterval);
            resolve();
          }
        }, 50);
      });
    }

    const categories = window.tagSidebarData.categories;
    const appCardsData = window.tagSidebarData.appCardsData;

    if (!categories || !appCardsData) {
      console.error("Required data could not be loaded");
      if (loading) loading.hide();
      return;
    }

    // Render sidebar only once
    renderSidebar(categories, appCardsData);

    // Handle initial URL parameters - only updates the highlights
    handleCategoryParamChange();
    // Handle browser back/forward navigation
    window.addEventListener('popstate', function(event) {
      handleCategoryParamChange();
    });

    // Handle window resize - only add the listener once
    if (!sidebarCache.initialized) {
      let resizeTimeout;
      window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function() {
          const sidebar = sidebarCache.sidebarElement || document.querySelector('.tag-sidebar.md-sidebar');
          if (sidebar) {
            sidebar.style.height = `calc(100vh - ${document.querySelector('.md-header').offsetHeight}px)`;
          }
        }, 100);
      });
      sidebarCache.initialized = true;
    }

    console.log("Tag sidebar ready");
    document.body.classList.add('tag-sidebar-ready');
    document.body.classList.add('content-ready');
    // Hide loading indicator
    if (loading) {
      loading.hide();
    }
  } catch (error) {
    console.error('Error initializing UI:', error);
    if (loading) loading.hide();
  }
}

// Main initialization on DOMContentLoaded - will execute if readystatechange hasn't triggered yet
document.addEventListener('DOMContentLoaded', function() {
  initializeIfNeeded();

  if (window.tagSidebarData && window.tagSidebarData.preloadComplete) {
    initializeUI();
  }
});

async function loadDataWithCache(dataPath, forceRefresh = false) {
  let data = {
    categories: null,
    appCardsData: null
  };

  // If preload has completed or is in progress, use those results
  if (window.tagSidebarData.preloadStarted) {
    console.log("Using preloaded data...");
    if (window.tagSidebarData.preloadComplete) {
      return {
        categories: window.tagSidebarData.categories,
        appCardsData: window.tagSidebarData.appCardsData
      };
    }
    // If preload is still in progress, wait for it to complete
    if (window.tagSidebarData.isLoading) {
      console.log("Waiting for preload to complete...");
      try {
        const [categories, appCards] = await Promise.all([
          window.tagSidebarData.categoriesPromise,
          window.tagSidebarData.appCardsPromise
        ]);
        return {
          categories: categories,
          appCardsData: appCards
        };
      } catch (error) {
        console.error("Error while waiting for preload:", error);
      }
    }
  }

  if (forceRefresh || !window.tagSidebarData.preloadComplete) {
    try {
      console.log("Fetching data from server with HTTP caching");

      const [categoriesData, appCardsData] = await Promise.all([
        fetchDataWithCache(`${dataPath}tmp_tag-categories.json`, forceRefresh),
        fetchDataWithCache(`${dataPath}tmp_app_cards.json`, forceRefresh)
      ]);

      data.categories = categoriesData;
      data.appCardsData = appCardsData;

      return data;
    } catch (error) {
      console.error("Error fetching data:", error);
      return data;
    }
  }

  // Fallback to whatever we have
  return {
    categories: window.tagSidebarData.categories,
    appCardsData: window.tagSidebarData.appCardsData
  };
}

// Global cache state to prevent duplicate loads
let dataLoadPromise = null;

function loadDataWithCacheShared(dataPath, forceRefresh = false) {
  if (dataLoadPromise) {
    return dataLoadPromise;
  }
  dataLoadPromise = loadDataWithCache(dataPath, forceRefresh);
  dataLoadPromise.finally(() => {
    setTimeout(() => {
      dataLoadPromise = null;
    }, 100);
  });

  return dataLoadPromise;
}

// Global reference to sidebar elements to avoid repeated querying
let sidebarCache = {
  initialized: false,
  sidebarElement: null,
  categoryItems: null,
  renderedCategories: false
};

// Function to create tag sidebar element
function createTagSidebarElement() {
  const tagSidebar = document.createElement('div');
  tagSidebar.className = 'tag-sidebar';
  return tagSidebar;
}

// Function to create and render the sidebar only once
function renderSidebar(categories, appCardsData) {
  // Skip if already rendered or no data
  if (sidebarCache.renderedCategories || !categories || !appCardsData) {
    return sidebarCache.sidebarElement;
  }

  console.log("Rendering sidebar structure...");

  let tagSidebar, primarySidebar;

  primarySidebar = document.querySelector('.md-sidebar--primary');
  if (primarySidebar) {
    console.log("Found primary sidebar");

    tagSidebar = primarySidebar.querySelector('.tag-sidebar');
    if (tagSidebar) {
      console.log("Tag sidebar already exists, skipping creation");
      sidebarCache.sidebarElement = tagSidebar;
      sidebarCache.renderedCategories = true;
      return tagSidebar;
    }

    tagSidebar = createTagSidebarElement();
    const scrollWrap = primarySidebar.querySelector('.md-sidebar__scrollwrap');
    if (scrollWrap) {
      scrollWrap.appendChild(tagSidebar);
      console.log("Sidebar inserted into primary sidebar scrollwrap");
    } else {
      // If no scrollwrap, create one
      const newScrollWrap = document.createElement('div');
      newScrollWrap.className = 'md-sidebar__scrollwrap';
      newScrollWrap.appendChild(tagSidebar);
      primarySidebar.appendChild(newScrollWrap);
      console.log("Created new scrollwrap and inserted sidebar");
    }
  } else {
    console.warn("Could not find primary sidebar, creating standalone");

    // Check if tag sidebar already exists
    tagSidebar = document.querySelector('.tag-sidebar.md-sidebar');
    if (tagSidebar) {
      console.log("Standalone tag sidebar already exists, skipping creation");
      sidebarCache.sidebarElement = tagSidebar;
      sidebarCache.renderedCategories = true;
      return tagSidebar;
    }

    // Create standalone sidebar if primary doesn't exist
    const mainInner = document.querySelector('.md-main__inner');
    if (!mainInner) {
      console.error("Could not find main inner container");
      return null;
    }

    // Create a wrapper for the sidebar
    const sidebarWrapper = document.createElement('div');
    sidebarWrapper.className = 'tag-sidebar-wrapper';
    sidebarWrapper.style.position = 'relative';

    tagSidebar = document.createElement('div');
    tagSidebar.className = 'tag-sidebar md-sidebar md-sidebar--primary';

    // Insert as first child
    if (mainInner.firstChild) {
      mainInner.insertBefore(sidebarWrapper, mainInner.firstChild);
    } else {
      mainInner.appendChild(sidebarWrapper);
    }
    sidebarWrapper.appendChild(tagSidebar);
    console.log("Created standalone sidebar");
  }

  // Build the sidebar content
  renderSidebarContent(tagSidebar, categories);

  if (primarySidebar) {
    primarySidebar.style.display = 'block';
  }

  // Cache references to avoid DOM queries later
  sidebarCache.sidebarElement = tagSidebar;
  sidebarCache.categoryItems = tagSidebar.querySelectorAll('.tag-category-item');
  sidebarCache.renderedCategories = true;

  document.body.classList.add('tag-sidebar-ready');

  console.log("Tag sidebar rendered");
  return tagSidebar;
}

// Function to render sidebar content
function renderSidebarContent(sidebarElement, categories) {
  const sidebarContent = document.createElement('div');
  sidebarContent.className = 'tag-sidebar-content';

  const title = document.createElement('h2');
  title.textContent = 'Application Categories';
  sidebarContent.appendChild(title);

  const categoryList = document.createElement('ul');
  categoryList.className = 'tag-category-list md-nav__list';

  const primaryCategories = categories;
  console.log("Primary categories:", primaryCategories.length);

  // Build the tags page URL with the correct base path
  const baseUrl = getBaseUrl();
  const tagsPath = `${baseUrl}tags/`;

  primaryCategories.forEach(category => {
    // Create category list item
    const categoryItem = createCategoryItem(category, tagsPath);
    categoryList.appendChild(categoryItem);
  });

  sidebarContent.appendChild(categoryList);
  sidebarElement.appendChild(sidebarContent);
}

// Function to create a category item element
function createCategoryItem(category, tagsPath) {
  const categoryItem = document.createElement('li');
  categoryItem.className = 'tag-category-item md-nav__item';
  categoryItem.dataset.category = category.title.toLowerCase();

  const appCount = category.count || 0;
  const categoryHeader = document.createElement('div');
  categoryHeader.className = 'tag-category-header md-nav__link';
  categoryHeader.innerHTML = `
    <span class="material-icons tag-category-icon">${category.icon}</span>
    <span class="tag-category-title">${category.title}</span>
    <span class="tag-category-count">(${appCount})</span>
  `;

  // Add event listeners
  categoryHeader.addEventListener('click', function(e) {
    e.preventDefault();
    const newUrl = `${tagsPath}?category=${encodeURIComponent(category.title)}`;
    window.location.href = newUrl;
  });

  categoryHeader.addEventListener('mouseenter', function() {
    this.classList.add('hover');
  });
  categoryHeader.addEventListener('mouseleave', function() {
    this.classList.remove('hover');
  });

  categoryItem.appendChild(categoryHeader);
  return categoryItem;
}

// Separate function to only update the active category highlight
function highlightActiveCategory(categoryName) {
  if (!categoryName || !sidebarCache.renderedCategories) return;

  // Only deal with DOM if sidebar is rendered
  if (!sidebarCache.categoryItems) {
    const sidebar = document.querySelector('.tag-sidebar');
    if (!sidebar) return;

    sidebarCache.categoryItems = sidebar.querySelectorAll('.tag-category-item');
    if (!sidebarCache.categoryItems.length) return;
  }

  const categoryLower = categoryName.toLowerCase();

  let activeItem = null;

  // Remove active class from all and find matching item
  for (const item of sidebarCache.categoryItems) {
    const header = item.querySelector('.tag-category-header');
    if (header) {
      if (item.dataset.category === categoryLower) {
        header.classList.add('active');
        activeItem = item;
      } else if (header.classList.contains('active')) {
        header.classList.remove('active');
      }
    }
  }

  // Scroll the matching item into view if found
  if (activeItem && sidebarCache.sidebarElement) {
    const sidebar = sidebarCache.sidebarElement;
    const itemTop = activeItem.offsetTop;
    const sidebarScrollTop = sidebar.scrollTop;
    const sidebarHeight = sidebar.clientHeight;

    // Only scroll if the item isn't fully visible
    if (itemTop < sidebarScrollTop || itemTop > sidebarScrollTop + sidebarHeight) {
      // Smooth scroll to position the item in the middle
      sidebar.scrollTo({
        top: itemTop - (sidebarHeight / 2),
        behavior: 'smooth'
      });
    }
  }
}

// Function to toggle category filter message visibility
function toggleCategoryFilterMessage(isVisible) {
  const filterMessage = document.querySelector('.category-filter-message');
  const resultsSection = document.querySelector('.category-results');
  if (filterMessage && resultsSection) {
    filterMessage.style.display = isVisible ? 'block' : 'none';
    resultsSection.style.display = isVisible ? 'none' : 'block';
  }
}

// Function to handle URL parameter changes - now just updates highlights
function handleCategoryParamChange() {
  const categoryParam = getUrlParams().get('category');

  if (categoryParam) {
    highlightActiveCategory(categoryParam);
    if (isTagsPage()) {
      loadCategoryContent(categoryParam);
    }
  } else if (isTagsPage()) {
    toggleCategoryFilterMessage(true);
  }
}

// Function to create or get the category content container
function getCategoryContentContainer() {
  let contentContainer = document.querySelector('.category-content');
  if (contentContainer) {
    return contentContainer;
  }

  // Create the container if it doesn't exist
  contentContainer = document.createElement('div');
  contentContainer.className = 'category-content';

  // Create the required child elements
  const filterMessage = document.createElement('div');
  filterMessage.className = 'category-filter-message';
  filterMessage.textContent = 'Select a category from the sidebar to view applications';

  const resultsSection = document.createElement('div');
  resultsSection.className = 'category-results';
  resultsSection.style.display = 'none';

  const cardsContainer = document.createElement('div');
  cardsContainer.className = 'category-cards';

  // Assemble the structure
  resultsSection.appendChild(cardsContainer);
  contentContainer.appendChild(filterMessage);
  contentContainer.appendChild(resultsSection);

  // Find the main content area and insert our container
  const mainContent = document.querySelector('.md-content__inner');
  if (mainContent) {
    mainContent.appendChild(contentContainer);
  } else {
    console.error("Could not find main content area");
    return null;
  }

  return contentContainer;
}

// Function to load category content without page reload
async function loadCategoryContent(category) {
  if (!category) return;

  console.log(`Loading content for category: ${category}`);

  try {
    // Get or create the content container
    const contentContainer = getCategoryContentContainer();
    if (!contentContainer) {
      return;
    }

    // Get references to message and results sections
    const filterMessage = contentContainer.querySelector('.category-filter-message');
    const resultsSection = contentContainer.querySelector('.category-results');
    const cardsContainer = contentContainer.querySelector('.category-cards');

    if (!filterMessage || !resultsSection || !cardsContainer) {
      console.error("Could not find required content elements");
      return;
    }

    // Add loading indicator
    contentContainer.classList.add('loading');
    toggleCategoryFilterMessage(false);
    cardsContainer.innerHTML = '<div class="loading-message">Loading applications...</div>';

    // Update page title
    document.title = `${category} - Applications - HoloHub`;

    // Highlight the active category in the sidebar
    highlightActiveCategory(category);

    // Use cached data from the global store
    const categoriesData = window.tagSidebarData.categories;
    const appCardsData = window.tagSidebarData.appCardsData || {};

    // Find matching category in categoriesData
    const matchingCategory = categoriesData.find(cat =>
      cat.title.toLowerCase() === category.toLowerCase()
    );

    if (!matchingCategory) {
      cardsContainer.innerHTML = '<p>No matching category found.</p>';
      contentContainer.classList.remove('loading');
      return;
    }

    renderCategoryContent(cardsContainer, matchingCategory, appCardsData);
  } catch (error) {
    console.error('Error loading category content:', error);
    const cardsContainer = document.querySelector('.category-cards');
    if (cardsContainer) {
      cardsContainer.innerHTML = `<p>Error loading applications: ${error.message}</p>`;
    }
  } finally {
    // Remove loading indicator
    const contentContainer = document.querySelector('.category-content');
    if (contentContainer) {
      contentContainer.classList.remove('loading');
    }
  }
}

// Function to render category content
function renderCategoryContent(container, matchingCategory, appCardsData) {
  // Filter apps based on the matching category title
  const categoryLower = matchingCategory.title.toLowerCase();
  const filteredApps = filterAppsByCategory(appCardsData, categoryLower);

  // Get base URL
  const baseUrl = getBaseUrl();

  // Display the results
  if (filteredApps.length === 0) {
    container.innerHTML = '<p>No applications found for this category.</p>';
    return;
  }

  container.innerHTML = '';

  // Add category header
  container.appendChild(createCategoryHeader(matchingCategory.title));

  // Create and append app grid with sorted app cards
  container.appendChild(createAppGrid(filteredApps, baseUrl));
}

// Create category header section
function createCategoryHeader(categoryTitle) {
  const categorySection = document.createElement('div');
  categorySection.className = 'category-section';

  const categoryHeader = document.createElement('div');
  categoryHeader.className = 'category-header';
  categoryHeader.innerHTML = `<h2 class="category-title">${categoryTitle}</h2>`;

  categorySection.appendChild(categoryHeader);
  return categorySection;
}

// Create grid with app cards
function createAppGrid(filteredApps, baseUrl) {
  // Create grid for cards
  const appGrid = document.createElement('div');
  appGrid.className = 'app-cards';

  // Sort apps alphabetically
  filteredApps.sort((a, b) => a[0].localeCompare(b[0]));

  // Create app cards
  filteredApps.forEach(([appName, appData]) => {
    const card = createAppCard(appName, appData.tags, appData, baseUrl);
    appGrid.appendChild(card);
  });

  return appGrid;
}

// Create a tag element with click handler
function createTagElement(tag) {
  const tagSpan = document.createElement('span');
  tagSpan.className = 'tag';
  tagSpan.textContent = tag;
  tagSpan.addEventListener('click', function(e) {
    e.stopPropagation(); // Prevent parent click events
    window.handleTagClick(tag);
    return false;
  });
  return tagSpan;
}

// Create tag count element with popup functionality
function createTagCountElement(tags) {
  const tagCount = document.createElement('span');
  tagCount.className = 'tag-count';
  tagCount.textContent = `+${tags.length - 3}`;
  tagCount.setAttribute('data-tags', JSON.stringify(tags));
  tagCount.addEventListener('click', function(e) {
    e.stopPropagation(); // Prevent parent click events
    if (typeof window.showAllTags === 'function') {
      window.showAllTags(this, this.getAttribute('data-tags'));
    }
    return false;
  });
  return tagCount;
}

// Create app thumbnail with placeholder and optional image
function createAppThumbnail(appName, cardData) {
  const simpleName = appName.split('/').pop();
  const thumbnail = document.createElement('div');
  thumbnail.className = 'app-thumbnail';

  // Generate a placeholder color based on app name
  const hash = appName.split('').reduce((a, b) => (((a << 5) - a) + b.charCodeAt(0))|0, 0);
  const hue = Math.abs(hash) % 360;
  const bgColor = `hsl(${hue}, 70%, 85%)`;

  // Get first letter of app title for placeholder
  const appInitial = (cardData.app_title || simpleName || appName).charAt(0).toUpperCase();

  // Add placeholder with app initial
  const placeholder = document.createElement('div');
  placeholder.className = 'image-placeholder';
  placeholder.style.backgroundColor = bgColor;
  placeholder.textContent = appInitial;
  thumbnail.appendChild(placeholder);

  // Add image if available
  if (cardData.image_url) {
    const img = document.createElement('img');
    img.src = cardData.image_url;
    img.alt = cardData.name || cardData.app_title;
    img.loading = 'lazy';
    img.onload = () => thumbnail.classList.add('loaded');
    thumbnail.appendChild(img);
  }

  return thumbnail;
}

// Method to create an app card with the enhanced tag count functionality
function createAppCard(appName, tags, cardData, baseUrl) {
  const card = document.createElement('div');
  card.className = 'app-card';
  const thumbnail = createAppThumbnail(appName, cardData);
  const details = document.createElement('div');
  details.className = 'app-details';
  details.innerHTML = `
    <h5>${cardData.app_title}</h5>
    <p>${cardData.description}</p>
  `;
  const tagsContainer = document.createElement('div');
  tagsContainer.className = 'app-tags';
  tags.slice(0, 3).forEach(tag => {
    tagsContainer.appendChild(createTagElement(tag));
  });
  if (tags.length > 3) {
    tagsContainer.appendChild(createTagCountElement(tags));
  }
  details.appendChild(tagsContainer);
  card.appendChild(thumbnail);
  card.appendChild(details);

  // Add click handler to navigate to app page
  card.addEventListener('click', (event) => {
    const url = `${baseUrl}${cardData.app_url}`;
    if (event.ctrlKey || event.metaKey) {
      // Open in new tab if Ctrl/Cmd key is pressed
      window.open(url, '_blank');
    } else {
      // Regular navigation
      window.location.href = url;
    }
  });
  card.style.cursor = 'pointer';

  return card;
}

// Function to filter apps by category
function filterAppsByCategory(appCardsData, categoryLower) {
  const categories = window.tagSidebarData.categories;
  const matchingCategory = categories.find(cat =>
    cat.title.toLowerCase() === categoryLower
  );
  if (!matchingCategory || !matchingCategory.ids) {
    return [];
  }
  const categoryIds = matchingCategory.ids.map(id => id.toLowerCase());
  return Object.entries(appCardsData)
    .filter(([appName, appData]) => {
      const appTitle = (appData.app_title || appName).toLowerCase();
      return categoryIds.some(id =>
        appTitle === id || appTitle.includes(id) || id.includes(appTitle)
      );
    });
}

// Global function for handling tag clicks across the site
window.handleTagClick = function(tag) {
  const searchInput = document.querySelector('.md-search__input');
  if (searchInput) {
    searchInput.focus();
    searchInput.value = tag;
    searchInput.dispatchEvent(new Event('input', { bubbles: true }));
    const searchButton = document.querySelector('[data-md-toggle="search"]');
    if (searchButton && !searchButton.checked) {
      searchButton.checked = true;
    }
  }
  return false;
};

// TagPopup class for managing tag popups
class TagPopup {
  constructor() {
    this.element = document.createElement('div');
    this.element.className = 'tags-popup';
    this.element.style.display = 'none';
    document.body.appendChild(this.element);
    document.addEventListener('click', this.handleOutsideClick.bind(this));
  }
  handleOutsideClick(e) {
    if (!this.element.contains(e.target) && !e.target.classList.contains('tag-count')) {
      this.hide();
    }
  }
  createPopupContent(tags) {
    this.element.innerHTML = '';

    // Add title
    const title = document.createElement('div');
    title.className = 'tags-popup-title';
    title.textContent = 'All Tags';
    this.element.appendChild(title);
    const content = document.createElement('div');
    content.className = 'tags-popup-content';
    this.element.appendChild(content);
    this.addTagsToContent(content, tags);
  }

  // Add tags to the content container
  addTagsToContent(content, tags) {
    tags.forEach(tag => {
      const tagEl = document.createElement('span');
      tagEl.className = 'tags-popup-tag';
      tagEl.textContent = tag;
      tagEl.addEventListener('click', () => {
        this.hide();
        window.handleTagClick(tag);
        return false;
      });
      content.appendChild(tagEl);
    });
  }

  show(targetElement, tags) {
    this.createPopupContent(tags);
    this.positionPopup(targetElement);
    this.element.style.display = 'block';
  }

  positionPopup(targetElement) {
    const rect = targetElement.getBoundingClientRect();
    this.element.style.top = (rect.bottom + window.scrollY + 8) + 'px';
    this.element.style.left = (rect.left + window.scrollX) + 'px';
  }

  hide() {
    this.element.style.display = 'none';
  }
}

// Initialize the global tag popup instance
let globalTagPopup;
