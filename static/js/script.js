document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictionForm");
    const button = document.getElementById("predictButton");

    if (form && button) {
        form.addEventListener("submit", function () {
            button.disabled = true;
            button.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2"></span>
                AI is analyzing...
            `;
        });
    }

    const cards = document.querySelectorAll(".feature-card, .stat-card, .metric-card, .content-card");
    cards.forEach((card, index) => {
        card.style.opacity = "0";
        card.style.transform = "translateY(12px)";
        setTimeout(() => {
            card.style.transition = "all 0.5s ease";
            card.style.opacity = "1";
            card.style.transform = "translateY(0)";
        }, 80 * index);
    });
});