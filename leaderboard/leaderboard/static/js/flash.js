document.addEventListener('DOMContentLoaded', () => {
    const defaultToastContainer = document.getElementById('toast-container-default');
    const defaultToastTemplates = document.querySelectorAll('.toast-template-default');

    if (defaultToastContainer) {
        defaultToastTemplates.forEach(template => {
            createToast(template, defaultToastContainer, 'default');
        });
    }

    const errorToastContainer = document.getElementById('toast-container-error');
    const errorToastTemplates = document.querySelectorAll('.toast-template-error');

    if (errorToastContainer) {
        errorToastTemplates.forEach(template => {
            createToast(template, errorToastContainer, 'error');
        });
    }

    function createToast(template, container, type) {
        const toast = document.createElement('div');
        const message = template.dataset.message;

        // Apply base and category-specific classes.
        toast.className = `toast-${type}`;
        toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
        toast.setAttribute('aria-live', type === 'error' ? 'assertive' : 'polite');

        // Apply the fade-in animation.
        toast.classList.add(`animate-${type}-toast-in`);

        toast.innerHTML = `
            <div class="toast-${type}-content">
                <div class="toast-${type}-message">
                    <p>${message}</p>
                </div>
                <div class="toast-${type}-close-btn">
                    <button type="button" class="toast-close-btn">
                        <span class="sr-only">Close</span>
                        <span aria-hidden="true">&#x2716;</span>
                    </button>
                </div>
            </div>
        `;

        // Add close functionality.
        const closeButton = toast.querySelector('.toast-close-btn');
        closeButton.addEventListener('click', () => {
            dismissToast(toast, type);
        });

        // Auto-dismiss timer (only for default messages).
        if (type === 'default') {
            const timerId = setTimeout(() => {
                dismissToast(toast, type);
            }, 5000); // Dismiss after 5 seconds

            // Store timerId to clear if closed manually
            toast.dataset.timerId = timerId;
        }

        // Prepend to container
        container.insertBefore(toast, container.firstChild);
    }

    function dismissToast(toast, type) {
        if (toast.dataset.timerId) {
            clearTimeout(parseInt(toast.dataset.timerId));
        }

        // Add fade-out animation class
        toast.classList.remove(`animate-${type}-toast-in`);
        toast.classList.add(`animate-${type}-toast-out`);

        // Remove the element after the animation completes
        toast.addEventListener('animationend', () => {
            if (toast.parentNode) {
                toast.remove();
            }
        });

        // Fallback removal in case animationend doesn't fire reliably
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 500);
    }
}); 