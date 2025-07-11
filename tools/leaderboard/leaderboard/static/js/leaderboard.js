// Toggles the visibility of the reference implementation code block.
document.addEventListener('DOMContentLoaded', function() {
    const referenceImpl = document.getElementById('referenceImpl');
    const toggleBtn = document.getElementById('toggleCodeBtn');

    let isExpanded = false;
    
    if (toggleBtn && referenceImpl) {
        const codeBlockFade = document.createElement('div');
        codeBlockFade.className = 'code-block-fade';
        referenceImpl.appendChild(codeBlockFade);

        toggleBtn.addEventListener('click', function() {
            if (isExpanded) {
                referenceImpl.classList.remove('max-h-none');
                referenceImpl.classList.add('max-h-[300px]');
                referenceImpl.classList.remove('overflow-y-auto');
                referenceImpl.classList.add('overflow-y-hidden');
                toggleBtn.textContent = 'Show';
                codeBlockFade.style.display = 'block';
                isExpanded = false;
            } else {
                referenceImpl.classList.remove('max-h-[300px]');
                referenceImpl.classList.add('max-h-none');
                referenceImpl.classList.remove('overflow-y-hidden');
                referenceImpl.classList.add('overflow-y-auto');
                toggleBtn.textContent = 'Hide';
                codeBlockFade.style.display = 'none';
                isExpanded = true;
            }
        });
    }
});

/**
 * Copies the code from the reference implementation code block to the
 * clipboard.
 */
function copyCode() {
    const codeElement = document.querySelector('#codeBlock');
    const textArea = document.createElement('textarea');
    textArea.value = codeElement.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    
    const copyButton = document.getElementById('copyCodeBtn');
    if (copyButton) {
        const originalText = copyButton.textContent;
        copyButton.textContent = 'Copied!';
        setTimeout(() => {
            copyButton.textContent = originalText;
        }, 2000);
    }
} 

/**
 * Toggles the visibility of the rankings for a given GPU type section.
 * @param {HTMLElement} button - The button element that was clicked.
 */
function toggleRankings(button) {
    const gpuId = button.getAttribute('data-gpu-id');
    const totalCount = parseInt(button.getAttribute('data-count'));
    const section = document.getElementById('section-' + gpuId);
    const rows = section.querySelectorAll('tr.rank-row');
    
    // Check if we're currently showing all (button text contains "Show Top 3")
    const isShowingAll = button.textContent.includes('Show Top 3');
    
    // Toggle visibility based on current state
    rows.forEach(row => {
        const rank = parseInt(row.getAttribute('data-rank'));
        if (rank > 3) {
            if (isShowingAll) {
                // Hide rows beyond top 3
                row.classList.add('hidden-row');
            } else {
                // Show all rows
                row.classList.remove('hidden-row');
            }
        }
    });
    
    // Update button text
    if (isShowingAll) {
        button.textContent = `Show All (${totalCount})`;
    } else {
        button.textContent = 'Show Top 3';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const copyBtn = document.getElementById('copyCodeBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', copyCode);
    }

    // Add listeners for ranking toggle buttons
    const rankingButtons = document.querySelectorAll('.rankings-btn');
    rankingButtons.forEach(button => {
        button.addEventListener('click', function() {
            toggleRankings(this); // Pass the clicked button element
        });
    });

    const referenceImpl = document.getElementById('referenceImpl');
    const toggleBtn = document.getElementById('toggleCodeBtn');

    let isExpanded = false;
    
    if (toggleBtn && referenceImpl) {
        const codeBlockFade = document.createElement('div');
        codeBlockFade.className = 'code-block-fade';
        // Only append fade if it doesn't exist
        if (!referenceImpl.querySelector('.code-block-fade')) {
             referenceImpl.appendChild(codeBlockFade);
        }

        toggleBtn.addEventListener('click', function() {
            const currentFade = referenceImpl.querySelector('.code-block-fade'); // Get fade element again
            if (isExpanded) {
                referenceImpl.classList.remove('max-h-none');
                referenceImpl.classList.add('max-h-[300px]');
                referenceImpl.classList.remove('overflow-y-auto');
                referenceImpl.classList.add('overflow-y-hidden');
                toggleBtn.textContent = 'Show';
                if(currentFade) currentFade.style.display = 'block'; // Check if fade exists
                isExpanded = false;
            } else {
                referenceImpl.classList.remove('max-h-[300px]');
                referenceImpl.classList.add('max-h-none');
                referenceImpl.classList.remove('overflow-y-hidden');
                referenceImpl.classList.add('overflow-y-auto');
                toggleBtn.textContent = 'Hide';
                 if(currentFade) currentFade.style.display = 'none'; // Check if fade exists
                isExpanded = true;
            }
        });
    }
});