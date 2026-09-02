document.addEventListener(
    "DOMContentLoaded",
    function () {

        const fileInput =
            document.getElementById(
                "knowledgeFile"
            );

        const uploadButton =
            document.getElementById(
                "uploadKnowledgeBtn"
            );

        const clearButton =
            document.getElementById(
                "clearKnowledgeBtn"
            );

        const loadingElement =
            document.getElementById(
                "knowledgeLoading"
            );

        const statusElement =
            document.getElementById(
                "knowledgeStatus"
            );

        const resultElement =
            document.getElementById(
                "knowledgeResult"
            );


        if (!fileInput || !uploadButton) {

            console.error(
                "Knowledge upload elements not found."
            );

            return;
        }


        // ======================================================
        // CSRF
        // ======================================================

        function getCSRFToken() {

            const csrfInput =
                document.querySelector(
                    "[name=csrfmiddlewaretoken]"
                );

            if (csrfInput) {

                return csrfInput.value;

            }

            return "";

        }


        // ======================================================
        // Upload
        // ======================================================

        uploadButton.addEventListener(
            "click",
            async function () {

                const file =
                    fileInput.files[0];


                if (!file) {

                    alert(
                        "Please select a file."
                    );

                    return;
                }


                // ------------------------------------------------
                // Client-side validation
                // ------------------------------------------------

                const allowedExtensions = [
                    ".pdf",
                    ".jpg",
                    ".csv",
                    ".xlsx"
                ];

                const filename =
                    file.name.toLowerCase();

                const extension =
                    filename.substring(
                        filename.lastIndexOf(".")
                    );


                if (
                    !allowedExtensions.includes(
                        extension
                    )
                ) {

                    alert(
                        "Only PDF, JPG, CSV and XLSX files are allowed."
                    );

                    return;
                }


                if (
                    file.size >
                    10 * 1024 * 1024
                ) {

                    alert(
                        "File size must not exceed 10 MB."
                    );

                    return;
                }


                // ------------------------------------------------
                // Loading
                // ------------------------------------------------

                uploadButton.disabled = true;

                fileInput.disabled = true;

                loadingElement.style.display =
                    "block";

                resultElement.innerHTML =
                    "";

                statusElement.innerHTML =
                    "Reading document...";


                // Give the UI a chance to render
                // before the long request begins.

                await new Promise(
                    function (resolve) {

                        setTimeout(
                            resolve,
                            100
                        );

                    }
                );


                statusElement.innerHTML =
                    "Creating embeddings and storing data in vector DB...";


                try {

                    const formData =
                        new FormData();

                    formData.append(
                        "file",
                        file
                    );


                    const response =
                        await fetch(
                            "/knowledge/upload/",
                            {
                                method: "POST",

                                headers: {

                                    "X-CSRFToken":
                                        getCSRFToken(),

                                },

                                body:
                                    formData,
                            }
                        );


                    const result =
                        await response.json();


                    if (!response.ok) {

                        throw new Error(
                            result.message ||
                            result.detail ||
                            "Upload failed."
                        );

                    }


                    if (
                        result.success === false
                    ) {

                        throw new Error(
                            result.message ||
                            "Unable to index document."
                        );

                    }


                    // ------------------------------------------------
                    // Success
                    // ------------------------------------------------

                    const data =
                        result.data || {};


                    statusElement.innerHTML =
                        "Document indexed successfully.";


                    resultElement.innerHTML = `

                        <div class="alert alert-success">

                            <strong>
                                Document indexed successfully.
                            </strong>

                            <br>

                            Document:
                            ${escapeHtml(
                                data.document_name ||
                                file.name
                            )}

                            <br>

                            Chunks created:
                            ${data.chunks_created || 0}

                        </div>

                    `;


                    // Clear selected file

                    fileInput.value = "";


                } catch (error) {

                    console.error(
                        "Knowledge upload error:",
                        error
                    );


                    statusElement.innerHTML =
                        "Indexing failed.";


                    resultElement.innerHTML = `

                        <div class="alert alert-danger">

                            <strong>
                                Unable to index document.
                            </strong>

                            <br>

                            ${escapeHtml(
                                error.message
                            )}

                        </div>

                    `;


                } finally {

                    uploadButton.disabled =
                        false;

                    fileInput.disabled =
                        false;

                    setTimeout(
                        function () {

                            loadingElement.style.display =
                                "none";

                        },
                        1500
                    );

                }

            }
        );


        if (clearButton) {

            clearButton.addEventListener(
                "click",
                async function () {

                    const confirmed = window.confirm(
                        "Delete all stored embeddings and vector data?\n\n" +
                        "New uploads will still be indexed the same way as now."
                    );

                    if (!confirmed) {
                        return;
                    }

                    clearButton.disabled = true;

                    if (uploadButton) {
                        uploadButton.disabled = true;
                    }

                    if (fileInput) {
                        fileInput.disabled = true;
                    }

                    loadingElement.style.display = "block";
                    resultElement.innerHTML = "";
                    statusElement.innerHTML =
                        "Deleting embeddings and vector data...";

                    try {

                        const response = await fetch(
                            "/knowledge/clear/",
                            {
                                method: "POST",
                                headers: {
                                    "X-CSRFToken": getCSRFToken(),
                                },
                            }
                        );

                        const result = await response.json();

                        if (!response.ok || result.success === false) {
                            throw new Error(
                                result.message ||
                                result.detail ||
                                "Unable to delete vector data."
                            );
                        }

                        const data = result.data || {};

                        statusElement.innerHTML =
                            "Vector data deleted.";

                        resultElement.innerHTML = `

                            <div class="alert alert-success">

                                <strong>
                                    Vector data deleted.
                                </strong>

                                <br>

                                Embeddings removed:
                                ${data.deleted_chunks || 0}

                                <br>

                                Tables removed:
                                ${data.deleted_tables || 0}

                            </div>

                        `;

                    } catch (error) {

                        console.error(
                            "Knowledge clear error:",
                            error
                        );

                        statusElement.innerHTML =
                            "Delete failed.";

                        resultElement.innerHTML = `

                            <div class="alert alert-danger">

                                <strong>
                                    Unable to delete vector data.
                                </strong>

                                <br>

                                ${escapeHtml(error.message)}

                            </div>

                        `;

                    } finally {

                        clearButton.disabled = false;

                        if (uploadButton) {
                            uploadButton.disabled = false;
                        }

                        if (fileInput) {
                            fileInput.disabled = false;
                        }

                        setTimeout(
                            function () {
                                loadingElement.style.display = "none";
                            },
                            1500
                        );

                    }

                }
            );

        }


        // ======================================================
        // Escape HTML
        // ======================================================

        function escapeHtml(value) {

            return String(value)

                .replaceAll(
                    "&",
                    "&amp;"
                )

                .replaceAll(
                    "<",
                    "&lt;"
                )

                .replaceAll(
                    ">",
                    "&gt;"
                )

                .replaceAll(
                    '"',
                    "&quot;"
                )

                .replaceAll(
                    "'",
                    "&#039;"
                );

        }

    }
);