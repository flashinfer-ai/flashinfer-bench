document.addEventListener("DOMContentLoaded", function () {
  // Smooth fade-in on load
  document.body.style.opacity = 0;
  document.body.style.transition = "opacity 0.3s ease-in-out";
  setTimeout(() => {
    document.body.style.opacity = 1;
  }, 10);

  // Optional: Scroll to top on internal link click
  document.querySelectorAll("a[href^='/']").forEach(link => {
    link.addEventListener("click", () => window.scrollTo(0, 0));
  });

  // Optional: Auto-dismiss flash messages if you use them
  const flash = document.getElementById("flash-message");
  if (flash) {
    setTimeout(() => {
      flash.style.opacity = "0";
      setTimeout(() => flash.remove(), 500);
    }, 3000);
  }
});
