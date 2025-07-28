function toggleSolution(id) {
  const el = document.getElementById(`solution-${id}`);
  if (el) {
    el.classList.toggle("hidden");
  }
}

function copyToClipboard(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const text = el.innerText;
  navigator.clipboard.writeText(text).then(() => {
    alert("Solution copied to clipboard!");
  });
}
