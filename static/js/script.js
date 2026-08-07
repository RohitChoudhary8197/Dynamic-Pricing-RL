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

    const revealItems = document.querySelectorAll(
        ".hero-title, .hero-description, .hero-buttons, .hero-tech, .ai-visual-card, .prediction-card, .feature-card, .stat-card, .metric-card, .content-card"
    );

    revealItems.forEach((item, index) => {
        item.style.opacity = "0";
        item.style.transform = "translateY(16px)";
        item.style.transition = "all 0.7s ease";
        setTimeout(() => {
            item.style.opacity = "1";
            item.style.transform = "translateY(0)";
        }, 90 * index);
    });
});