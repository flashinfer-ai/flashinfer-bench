document.addEventListener("DOMContentLoaded", function () {
  // Toggle show/hide
  document.querySelectorAll(".toggle-solution").forEach((btn) => {
    btn.addEventListener("click", function () {
      const targetId = btn.getAttribute("data-target");
      const targetRow = document.getElementById(targetId);
      const isHidden = targetRow.classList.contains("hidden");

      if (isHidden) {
        targetRow.classList.remove("hidden");
        btn.textContent = "Hide";
      } else {
        targetRow.classList.add("hidden");
        btn.textContent = "Show";
      }
    });
  });

  // Copy to clipboard
  document.querySelectorAll(".copy-button").forEach((btn) => {
    btn.addEventListener("click", function () {
      const codeId = btn.getAttribute("data-target");
      const codeBlock = document.getElementById(codeId);
      const text = codeBlock?.innerText || "";

      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = "Copied!";
        setTimeout(() => {
          btn.textContent = "Copy to Clipboard";
        }, 2000);
      });
    });
  });

  document.getElementById("workload-select").addEventListener("change", function () {
    const selected = this.value;
    document.querySelectorAll(".workload-section").forEach(section => {
      const workload = section.getAttribute("data-workload");
      if (selected === "all" || workload === selected) {
        section.style.display = "block";
      } else {
        section.style.display = "none";
      }
    });
  });
});
