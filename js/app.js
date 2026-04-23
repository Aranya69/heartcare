/**
 * HeartCare AI — Frontend JavaScript
 * Handles AJAX predictions, sidebar toggle, form interactions, and UI animations.
 */

// ─── Sidebar Toggle ───
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        sidebar.classList.toggle('open');
    }
}

// Close sidebar when clicking outside on mobile
document.addEventListener('click', function(e) {
    const sidebar = document.getElementById('sidebar');
    const toggle = document.getElementById('menuToggle');
    if (sidebar && sidebar.classList.contains('open') && 
        !sidebar.contains(e.target) && !toggle.contains(e.target)) {
        sidebar.classList.remove('open');
    }
});

// ─── Flash Message Auto-dismiss ───
document.addEventListener('DOMContentLoaded', function() {
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(function(msg) {
        setTimeout(function() {
            msg.style.opacity = '0';
            msg.style.transform = 'translateY(-10px)';
            setTimeout(function() { msg.remove(); }, 300);
        }, 5000);
    });
});

// ─── Predict Form AJAX Submission ───
document.addEventListener('DOMContentLoaded', function() {
    const predictForm = document.getElementById('predictForm');
    if (!predictForm) return;

    // Auto-fill age and sex from selected patient
    const patientSelect = document.getElementById('patient_id');
    if (patientSelect) {
        patientSelect.addEventListener('change', function() {
            const selectedOption = this.options[this.selectedIndex];
            if (selectedOption && selectedOption.value) {
                const text = selectedOption.text;
                const ageMatch = text.match(/\((\d+)y/);
                const sexMatch = text.match(/(Male|Female)\)/);
                
                if (ageMatch) {
                    document.getElementById('age').value = ageMatch[1];
                }
                if (sexMatch) {
                    document.getElementById('sex').value = sexMatch[1] === 'Male' ? 'M' : 'F';
                }
            }
        });
    }

    predictForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const formData = new FormData(predictForm);
        const submitBtn = document.getElementById('predictBtn');
        const originalText = submitBtn.innerHTML;

        // Show loading state
        submitBtn.innerHTML = '⏳ Analyzing...';
        submitBtn.disabled = true;

        fetch(predictForm.action, {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(function(response) { return response.json(); })
        .then(function(data) {
            if (data.success) {
                showResult(data);
            } else {
                alert('Prediction error: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(function(error) {
            console.error('Error:', error);
            alert('Network error. Please try again.');
        })
        .finally(function() {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        });
    });
});

// ─── Show Prediction Result ───
function showResult(data) {
    const resultPanel = document.getElementById('resultPanel');
    const infoPanel = document.getElementById('infoPanel');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const probBar = document.getElementById('probBar');
    const probText = document.getElementById('probText');

    if (!resultPanel) return;

    // Hide info panel, show result
    if (infoPanel) infoPanel.style.display = 'none';
    resultPanel.style.display = 'block';

    if (data.result === 1) {
        // Heart Disease Detected
        resultIcon.textContent = '⚠️';
        resultTitle.textContent = 'Heart Disease Detected';
        resultTitle.className = 'result-title result-disease';
        probBar.className = 'prob-bar prob-danger';
        probBar.style.width = data.probability + '%';
        probText.textContent = 'Disease Probability: ' + data.probability + '%';
    } else {
        // Healthy
        resultIcon.textContent = '✅';
        resultTitle.textContent = 'Patient Appears Healthy';
        resultTitle.className = 'result-title result-healthy';
        probBar.className = 'prob-bar prob-safe';
        probBar.style.width = (100 - data.probability) + '%';
        probText.textContent = 'Health Confidence: ' + (100 - data.probability).toFixed(1) + '%';
    }

    // Scroll to result
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Reset Form ───
function resetForm() {
    const predictForm = document.getElementById('predictForm');
    const resultPanel = document.getElementById('resultPanel');
    const infoPanel = document.getElementById('infoPanel');

    if (predictForm) {
        predictForm.reset();
    }
    if (resultPanel) {
        resultPanel.style.display = 'none';
    }
    if (infoPanel) {
        infoPanel.style.display = 'block';
    }

    // Scroll back to form
    if (predictForm) {
        predictForm.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ─── Register Form Validation ───
document.addEventListener('DOMContentLoaded', function() {
    const registerForm = document.getElementById('registerForm');
    if (!registerForm) return;

    registerForm.addEventListener('submit', function(e) {
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirm_password').value;

        if (password !== confirmPassword) {
            e.preventDefault();
            alert('Passwords do not match!');
        }
    });
});
