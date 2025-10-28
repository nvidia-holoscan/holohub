/**
 * Run Locally Button - Dynamic Modal Creation and Copy Functionality
 * Modal is created entirely in JavaScript to avoid search indexing
 */

document.addEventListener('DOMContentLoaded', function() {
    const COPY_ICON_SVG = '<svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
    
    // Function to create modal HTML
    function createModal(appName, appShortName, appLanguage, needsLanguageFlag) {
        // Create commands - add --language cpp only if both Python and C++ versions exist
        const cmd1 = 'git clone https://github.com/nvidia-holoscan/holohub.git';
        const languageFlag = (needsLanguageFlag === 'true') ? ' --language cpp' : '';
        const cmd2 = `./holohub run ${appShortName}${languageFlag}`;
        // For "Copy All", include cd holohub
        const allCommands = `${cmd1} && cd holohub && ${cmd2}`;
        
        // Create step 2 label with language
        const step2Label = appLanguage 
            ? `Step 2: Run the ${appLanguage} application` 
            : 'Step 2: Run the application';
        
        // Create modal element
        const modal = document.createElement('div');
        modal.className = 'run-locally-modal';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.setAttribute('aria-labelledby', 'run-locally-title');
        
        modal.innerHTML = `
            <div class="run-locally-modal-content">
                <div class="run-locally-modal-header">
                    <h3 id="run-locally-title">Run ${appName} Locally</h3>
                    <button class="run-locally-close" aria-label="Close">&times;</button>
                </div>
                <button class="run-locally-copy-all-btn" data-command="${allCommands}">
                    ${COPY_ICON_SVG}
                    Copy All
                </button>
                <div class="run-locally-commands-section">
                    <div class="run-locally-command-container">
                        <div class="run-locally-command-label">Step 1: Clone the repository</div>
                        <pre class="run-locally-command">${cmd1}</pre>
                        <button class="run-locally-copy-btn" data-command="${cmd1}">
                            ${COPY_ICON_SVG}
                            Copy
                        </button>
                    </div>
                    <div class="run-locally-command-container">
                        <div class="run-locally-command-label">${step2Label}</div>
                        <pre class="run-locally-command">${cmd2}</pre>
                        <button class="run-locally-copy-btn" data-command="${cmd2}">
                            ${COPY_ICON_SVG}
                            Copy
                        </button>
                    </div>
                </div>
                <p class="run-locally-description">
                    These commands will clone the HoloHub repository and run the application using the <code>holohub</code> CLI tool.
                </p>
            </div>
        `;
        
        return modal;
    }
    
    // Function to show modal
    function showModal(modal) {
        document.body.appendChild(modal);
        // Force reflow before setting display to trigger animation
        modal.offsetHeight;
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        
        // Setup event listeners for this modal
        setupModalEventListeners(modal);
    }
    
    // Function to close and remove modal
    function closeModal(modal) {
        modal.style.display = 'none';
        document.body.style.overflow = '';
        // Remove from DOM after animation
        setTimeout(() => {
            if (modal.parentNode) {
                modal.parentNode.removeChild(modal);
            }
        }, 200);
    }
    
    // Setup event listeners for a modal
    function setupModalEventListeners(modal) {
        // Close button
        const closeBtn = modal.querySelector('.run-locally-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => closeModal(modal));
        }
        
        // Click outside to close
        modal.addEventListener('click', function(event) {
            if (event.target === modal) {
                closeModal(modal);
            }
        });
        
        // Copy buttons
        const copyButtons = modal.querySelectorAll('.run-locally-copy-btn, .run-locally-copy-all-btn');
        copyButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                const command = this.getAttribute('data-command');
                copyToClipboard(command, this);
            });
        });
    }
    
    // Copy to clipboard function
    function copyToClipboard(text, button) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(() => {
                showCopySuccess(button);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
                fallbackCopyTextToClipboard(text, button);
            });
        } else {
            fallbackCopyTextToClipboard(text, button);
        }
    }
    
    // Show copy success feedback
    function showCopySuccess(button) {
        const originalHTML = button.innerHTML;
        const icon = button.querySelector('.copy-icon');
        const iconHTML = icon ? icon.outerHTML : '';
        
        button.innerHTML = iconHTML + 'Copied!';
        button.classList.add('copied');
        
        setTimeout(() => {
            button.innerHTML = originalHTML;
            button.classList.remove('copied');
        }, 2000);
    }
    
    // Fallback copy method for older browsers
    function fallbackCopyTextToClipboard(text, button) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.top = '0';
        textArea.style.left = '0';
        textArea.style.width = '2em';
        textArea.style.height = '2em';
        textArea.style.padding = '0';
        textArea.style.border = 'none';
        textArea.style.outline = 'none';
        textArea.style.boxShadow = 'none';
        textArea.style.background = 'transparent';
        
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            const successful = document.execCommand('copy');
            if (successful) {
                showCopySuccess(button);
            }
        } catch (err) {
            console.error('Fallback: Unable to copy', err);
        }
        
        document.body.removeChild(textArea);
    }
    
    // Handle Run Locally button clicks
    const runLocallyButtons = document.querySelectorAll('.run-locally-button');
    runLocallyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const appName = this.getAttribute('data-app-name');
            const appShortName = this.getAttribute('data-app-short-name');
            const appLanguage = this.getAttribute('data-app-language');
            const needsLanguageFlag = this.getAttribute('data-needs-language-flag');
            
            // Create and show modal
            const modal = createModal(appName, appShortName, appLanguage, needsLanguageFlag);
            showModal(modal);
        });
    });
    
    // Handle Escape key to close any open modal
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            const openModals = document.querySelectorAll('.run-locally-modal[style*="display: block"]');
            openModals.forEach(modal => closeModal(modal));
        }
    });
});
