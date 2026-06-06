(function () {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const dropZoneContent = document.getElementById("dropZoneContent");
    const imagePreview = document.getElementById("imagePreview");
    const removeImageBtn = document.getElementById("removeImage");
    const submitBtn = document.getElementById("submitBtn");

    if (!dropZone || !fileInput) return;

    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add("dragover"), false);
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove("dragover"), false);
    });

    dropZone.addEventListener("drop", handleDrop, false);
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", handleFiles);
    removeImageBtn.addEventListener("click", clearPreview);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFiles();
        }
    }

    function handleFiles() {
        const file = fileInput.files[0];
        if (!file) return;

        if (!file.type.startsWith("image/")) {
            flash("Format file tidak didukung. Pilih file gambar.", "danger");
            return;
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            dropZoneContent.classList.add("d-none");
            previewContainer.classList.remove("d-none");
            dropZone.classList.add("has-image");
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearPreview(e) {
        e.preventDefault();
        e.stopPropagation();
        fileInput.value = "";
        imagePreview.src = "";
        dropZoneContent.classList.remove("d-none");
        previewContainer.classList.add("d-none");
        dropZone.classList.remove("has-image");
        submitBtn.disabled = true;
    }

    const form = document.getElementById("uploadForm");
    if (form) {
        form.addEventListener("submit", function () {
            if (submitBtn && !submitBtn.disabled) {
                submitBtn.innerHTML =
                    '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Menganalisis...';
                submitBtn.disabled = true;
            }
        });
    }

    function flash(message, category) {
        const alerts = document.querySelectorAll(".alert");
        alerts.forEach((a) => a.remove());
        const alertDiv = document.createElement("div");
        alertDiv.className = `alert alert-${category} alert-dismissible fade show`;
        alertDiv.setAttribute("role", "alert");
        const icons = { danger: "exclamation-triangle-fill", success: "check-circle-fill", warning: "exclamation-circle-fill", info: "info-circle-fill" };
        alertDiv.innerHTML = `
            <i class="bi bi-${icons[category] || "info-circle-fill"} me-2"></i>${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        const main = document.querySelector("main");
        if (main && main.firstChild) {
            main.insertBefore(alertDiv, main.firstChild);
        } else {
            document.body.prepend(alertDiv);
        }
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alertDiv);
            if (bsAlert) bsAlert.close();
        }, 5000);
    }

    document.querySelectorAll(".alert .btn-close").forEach((btn) => {
        btn.addEventListener("click", function () {
            setTimeout(() => {
                const alert = this.closest(".alert");
                if (alert) alert.remove();
            }, 300);
        });
    });
})();
