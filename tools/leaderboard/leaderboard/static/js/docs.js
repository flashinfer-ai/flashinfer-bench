document.addEventListener("DOMContentLoaded", function () {
    const content = document.querySelector('.docs-content');
    const toc = document.getElementById('toc');

    if (content && toc) {
        const headers = content.querySelectorAll('h2, h3');
        let currentLevel = 2;
        let tocHtml = '<ul>';

        headers.forEach((header) => {
            const level = parseInt(header.tagName.substring(1));
            const id = header.innerText.replace(/\s+/g, '-').toLowerCase();
            header.id = id;

            if (level > currentLevel) {
                tocHtml += '<ul>';
            } else if (level < currentLevel) {
                tocHtml += '</ul>';
            }
            currentLevel = level;

            tocHtml += `<li><a href="#${id}" class="toc-link">${header.innerText}</a></li>`;
        });

        tocHtml += '</ul>';
        toc.innerHTML = tocHtml;

        // Scroll spy
        const tocLinks = toc.querySelectorAll('a');
        window.addEventListener('scroll', () => {
            let current = '';
            headers.forEach((header) => {
                const rect = header.getBoundingClientRect();
                if (rect.top <= 120) {
                    current = header.id;
                }
            });
            tocLinks.forEach((link) => {
                link.classList.toggle('active', link.getAttribute('href') === '#' + current);
            });
        });
    }
});