document.addEventListener('DOMContentLoaded', function () {
  // Event delegation for toggle solution buttons
  document.addEventListener('click', function (e) {
    if (e.target.classList.contains('toggle-solution')) {
      e.preventDefault();

      const targetId = e.target.getAttribute('data-target');

      const targetElement = document.getElementById(targetId);
      const button = e.target;

      if (targetElement) {
        const isHidden = targetElement.classList.contains('hidden');
        targetElement.classList.toggle('hidden');
        button.textContent = isHidden ? 'Hide' : 'Show';
      }
    }

    // Event delegation for copy buttons
    if (e.target.classList.contains('copy-button')) {
      e.preventDefault();

      const targetId = e.target.getAttribute('data-target');
      const codeElement = document.getElementById(targetId);

      if (codeElement) {
        const textarea = document.createElement('textarea');
        textarea.value = codeElement.textContent;
        document.body.appendChild(textarea);
        textarea.select();

        try {
          document.execCommand('copy');
          const originalText = e.target.textContent;
          e.target.textContent = 'Copied!';
          e.target.classList.add('text-green-600');

          setTimeout(() => {
            e.target.textContent = originalText;
            e.target.classList.remove('text-green-600');
          }, 2000);
        } catch (err) {
          console.error('Failed to copy text: ', err);
          const originalText = e.target.textContent;
          e.target.textContent = 'Copy failed';
          e.target.classList.add('text-red-600');

          setTimeout(() => {
            e.target.textContent = originalText;
            e.target.classList.remove('text-red-600');
          }, 2000);
        }

        document.body.removeChild(textarea);
      }
    }
  });

  // Workload filter functionality
  const workloadSelect = document.getElementById('workload-select');
  if (workloadSelect) {
    workloadSelect.addEventListener('change', function () {
      const selectedWorkload = this.value;
      const workloadSections = document.querySelectorAll('.workload-section');

      workloadSections.forEach(section => {
        const workloadName = section.getAttribute('data-workload');
        if (selectedWorkload === 'all' || workloadName === selectedWorkload) {
          section.classList.remove('hidden');
        } else {
          section.classList.add('hidden');
        }
      });

        // Hide all visible solutions
        document.querySelectorAll('.workload-section:not(.hidden) tr[id^="solution-"]:not(.hidden)')
        .forEach(row => row.classList.add('hidden'));

        // Reset all toggle buttons
        document.querySelectorAll('.workload-section:not(.hidden) .toggle-solution')
        .forEach(btn => btn.textContent = 'Show');
    });
  }
});