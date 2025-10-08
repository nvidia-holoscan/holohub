// Shared JavaScript for filtering cards on Applications and Benchmarks pages

function nat_at_top() {
    var navtab = document.getElementsByClassName('md-tabs');
    if (navtab && navtab[0]) {
        navtab[0].classList.add("topped");
    }
}

window.onload = nat_at_top;

window.addEventListener('scroll', function() {
    var scroll = document.body.scrollTop || document.documentElement.scrollTop;
    var navtab = document.getElementsByClassName('md-tabs');
    if (navtab && navtab[0]) {
        navtab[0].classList.toggle("topped", scroll < navtab[0].offsetHeight);
    }
});

// Generic filter function that can be customized per page
function createFilterFunction(options) {
    options = options || {};
    var normalizeSpaces = options.normalizeSpaces !== undefined ? options.normalizeSpaces : false;
    var updateTitle = options.updateTitle !== undefined ? options.updateTitle : false;
    var allTitle = options.allTitle || 'All';
    var navSelector = options.navSelector || 'nav a';
    var tagMatchStrategy = options.tagMatchStrategy || 'href'; // 'href' or 'text'
    
    return function filterByTag(tag) {
        var cards = document.querySelectorAll('.feature-box');
        var activeCategory = tag.toLowerCase();
        
        if (normalizeSpaces) {
            activeCategory = activeCategory.replace(/ /g, '-');
        }
        
        console.log('Filtering by tag:', tag, '(normalized:', activeCategory + ')');
        
        // Update the category title if enabled
        if (updateTitle) {
            var categoryTitle = document.getElementById('category-title');
            if (categoryTitle) {
                if (tag.toLowerCase() === 'all') {
                    categoryTitle.textContent = allTitle;
                } else {
                    categoryTitle.textContent = tag;
                }
            }
        }
        
        // If "all" is selected, show all cards
        if (activeCategory === 'all') {
            cards.forEach(function(card) {
                card.style.display = '';
            });
        } else {
            var shownCount = 0;
            // Filter cards by tag
            cards.forEach(function(card) {
                // Get all tags in the card
                var cardTags = card.querySelectorAll('.md-tag');
                var hasTag = false;
                
                // Check if any of the card's tags match the selected category
                cardTags.forEach(function(tagElement) {
                    if (tagMatchStrategy === 'href') {
                        // Match by href attribute (benchmarks style)
                        var tagHref = tagElement.getAttribute('href');
                        if (tagHref && tagHref.includes('tag:' + activeCategory)) {
                            hasTag = true;
                        }
                    } else if (tagMatchStrategy === 'text') {
                        // Match by text content (applications style)
                        var tagText = tagElement.textContent || tagElement.innerText;
                        var tagTextNormalized = tagText.toLowerCase().trim();
                        
                        if (normalizeSpaces) {
                            tagTextNormalized = tagTextNormalized.replace(/ /g, '-');
                        }
                        
                        if (tagTextNormalized === activeCategory) {
                            hasTag = true;
                            console.log('Match found:', tagText, 'matches', tag);
                        }
                    }
                });
                
                // Show or hide the card based on whether it has the tag
                if (hasTag) {
                    card.style.display = '';
                    shownCount++;
                } else {
                    card.style.display = 'none';
                }
            });
            console.log('Shown', shownCount, 'cards with tag:', tag);
        }
        
        // Update active state on navigation links
        var navLinks = document.querySelectorAll(navSelector);
        navLinks.forEach(function(link) {
            var isActive = false;
            
            // Check if this link matches the active category
            if (tagMatchStrategy === 'text') {
                // For applications: extract tag from onclick attribute
                var linkOnclick = link.getAttribute('onclick');
                var match = linkOnclick && linkOnclick.match(/filterByTag\('([^']+)'\)/);
                var linkTag = match ? match[1].toLowerCase() : '';
                
                if (normalizeSpaces) {
                    linkTag = linkTag.replace(/ /g, '-');
                }
                
                isActive = linkTag === activeCategory || (activeCategory === 'all' && link.getAttribute('href') === '#all');
            } else {
                // For benchmarks: match by href
                isActive = link.getAttribute('href') === '#' + activeCategory;
            }
            
            if (isActive) {
                link.style.backgroundColor = '#76b900';
                link.style.color = 'white';
                link.setAttribute('data-active', 'true');
            } else {
                link.style.backgroundColor = '';
                link.style.color = 'var(--md-default-fg-color)';
                link.removeAttribute('data-active');
            }
        });
    };
}

// Show all cards
function showAllCards(navSelector) {
    navSelector = navSelector || 'nav a';
    
    var cards = document.querySelectorAll('.feature-box');
    cards.forEach(function(card) {
        card.style.display = '';
    });
    
    // Remove active state from all navigation links
    var navLinks = document.querySelectorAll(navSelector);
    navLinks.forEach(function(link) {
        link.style.backgroundColor = '';
        link.style.color = 'var(--md-default-fg-color)';
        link.removeAttribute('data-active');
    });
}

// Setup hash change and page load handlers
function setupFilterHandlers(filterFn, navSelector) {
    // Handle hash change
    window.addEventListener('hashchange', function() {
        var hash = window.location.hash.substring(1); // Remove the #
        if (hash) {
            filterFn(hash);
        } else {
            showAllCards(navSelector);
        }
    });

    // Check for hash on page load
    window.addEventListener('load', function() {
        var hash = window.location.hash.substring(1);
        if (hash) {
            filterFn(hash);
        }
    });
}

