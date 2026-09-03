document.addEventListener("DOMContentLoaded", function () {

    const askButton =
        document.getElementById("askBtn");

    const questionInput =
        document.getElementById("question");

    const answerElement =
        document.getElementById("answer");

    const sourcesElement =
        document.getElementById("sources");

    const providerElement =
        document.getElementById("provider");

    const modelElement =
        document.getElementById("model");

    const confidenceElement =
        document.getElementById("confidence");

    const timeElement =
        document.getElementById("time");

    const tokensElement =
        document.getElementById("tokens");

        const costElement =
            document.getElementById("cost");

        const loadingElement =
            document.getElementById("loading");

        const clarificationPanel =
            document.getElementById("clarificationPanel");

        const clarificationText =
            document.getElementById("clarificationText");

        const clarificationOptions =
            document.getElementById("clarificationOptions");

        const newConversationBtn =
            document.getElementById("newConversationBtn");

        const responseStatus =
            document.getElementById("responseStatus");

        const runStatus =
            document.getElementById("runStatus");

        const runIntent =
            document.getElementById("runIntent");

        const resolvedQuery =
            document.getElementById("resolvedQuery");

    const webSearchSection =
        document.getElementById("webSearchSection");

    const webSearchBtn =
        document.getElementById("webSearchBtn");

    const webSearchLoading =
        document.getElementById("webSearchLoading");


    // ==========================================================
    // Validate elements
    // ==========================================================

    if (!askButton) {

        console.error(
            "askBtn not found."
        );

        return;
    }

    if (!questionInput) {

        console.error(
            "question input not found."
        );

        return;
    }


    // ==========================================================
    // CSRF
    // ==========================================================

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


    function getConversationId() {
        const key = "rag_conversation_id";
        let value = sessionStorage.getItem(key);
        if (!value) {
            value = (
                crypto.randomUUID &&
                crypto.randomUUID()
            ) || String(Date.now());
            sessionStorage.setItem(key, value);
        }
        return value;
    }


    function resetConversation() {
        sessionStorage.removeItem("rag_conversation_id");
        getConversationId();
        if (clarificationPanel) {
            clarificationPanel.style.display = "none";
        }
        if (clarificationOptions) {
            clarificationOptions.innerHTML = "";
        }
        if (questionInput) {
            questionInput.placeholder =
                "Example: What is happening at the Qatalum smelter?";
        }
    }


    if (newConversationBtn) {
        newConversationBtn.addEventListener(
            "click",
            resetConversation
        );
    }


    // ==========================================================
    // Ask button
    // ==========================================================

    askButton.addEventListener(
        "click",
        askQuestion
    );
    if (webSearchBtn) {

    webSearchBtn.addEventListener(
        "click",
        refineWithWebSearch
    );

}


    // ==========================================================
    // Ask Question
    // ==========================================================

    async function askQuestion() {

        const question =
            questionInput.value.trim();


        if (!question) {

            alert(
                "Please enter a question."
            );

            return;
        }


        // ------------------------------------------------------
        // Loading
        // ------------------------------------------------------

        askButton.disabled = true;

        askButton.innerHTML =
            "Generating...";


        if (loadingElement) {

            loadingElement.style.display =
                "block";

        }


        answerElement.innerHTML =
            "Generating answer...";


        // ------------------------------------------------------
        // Reset Execution Summary
        // ------------------------------------------------------

        providerElement.innerHTML = "-";

        modelElement.innerHTML = "-";

        confidenceElement.innerHTML = "-";

        timeElement.innerHTML = "-";

        tokensElement.innerHTML = "-";

        costElement.innerHTML = "-";

        if (runStatus) {
            runStatus.innerHTML = "-";
        }
        if (runIntent) {
            runIntent.innerHTML = "-";
        }
        if (resolvedQuery) {
            resolvedQuery.innerHTML = "-";
        }
        if (responseStatus) {
            responseStatus.innerHTML = "";
        }
        if (clarificationPanel) {
            clarificationPanel.style.display = "none";
        }


        sourcesElement.innerHTML =
            "Retrieving sources...";


        try {

            // ==================================================
            // Django API
            // ==================================================

            const response = await fetch(
                "/api/chat/",
                {
                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json",

                        "X-CSRFToken":
                            getCSRFToken(),

                    },

                    body: JSON.stringify({

                        question: question,
                        conversation_id: getConversationId()

                    }),

                }
            );


            // ==================================================
            // HTTP Error
            // ==================================================

            if (!response.ok) {

                const errorText =
                    await response.text();

                console.error(
                    "Chat API failed:",
                    response.status,
                    errorText
                );

                throw new Error(
                    `Request failed: ${response.status}`
                );
            }


            // ==================================================
            // Parse JSON
            // ==================================================

            const result =
                await response.json();


            console.log(
                "Django chat response:",
                result
            );


            // ==================================================
            // Application Error
            // ==================================================

            if (
                result.success === false
            ) {

                throw new Error(
                    result.message ||
                    "Unable to process request."
                );
            }


            // ==================================================
            // Extract RAG Result
            // ==================================================

            const data =
                result.data || result;


            console.log(
                "RAG response:",
                data
            );


            // ==================================================
            // Answer
            // ==================================================

            const answer =
    (data.answer || "").trim();

answerElement.innerHTML =
    answer ||
    "No answer returned.";


const statusValue =
    (data.status || "SUCCESS").toString();

const clarificationNeeded =
    data.clarification_required === true ||
    statusValue === "CLARIFICATION_REQUIRED";

if (runStatus) {
    runStatus.innerHTML = statusValue;
}
if (runIntent) {
    runIntent.innerHTML = data.intent || "-";
}
if (resolvedQuery) {
    resolvedQuery.innerHTML =
        data.resolved_question || question;
}

if (responseStatus) {
    let pillClass = "success";
    if (clarificationNeeded) {
        pillClass = "clarify";
    } else if (
        statusValue !== "SUCCESS"
    ) {
        pillClass = "warn";
    }
    responseStatus.innerHTML =
        `<span class="status-pill ${pillClass}">${escapeHtml(statusValue)}</span>`;
}

if (clarificationNeeded && clarificationPanel) {
    clarificationPanel.style.display = "block";
    if (clarificationText) {
        clarificationText.innerHTML =
            escapeHtml(
                data.clarification_question ||
                answer
            );
    }
    if (clarificationOptions) {
        clarificationOptions.innerHTML = "";
        const options = data.clarification_options || [];
        options.forEach(function (option) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "clarification-chip";
            button.innerText = option;
            button.addEventListener("click", function () {
                questionInput.value = option;
                askQuestion();
            });
            clarificationOptions.appendChild(button);
        });
    }
    questionInput.placeholder =
        "Type your clarification or tap an option above";
    questionInput.value = "";
} else if (clarificationPanel) {
    clarificationPanel.style.display = "none";
}


// ==========================================================
// Check whether RAG has enough information
// ==========================================================

const noKnowledgeMessage =
    "I don't have enough information in my knowledge base.";

const hasKnowledge =
    !clarificationNeeded &&
    answer.toLowerCase() !==
    noKnowledgeMessage.toLowerCase() &&
    !answer.toLowerCase().includes("couldn't find") &&
    !answer.toLowerCase().includes("can't reliably") &&
    statusValue === "SUCCESS";


// ==========================================================
// Web Search + Sources
// ==========================================================

if (hasKnowledge) {

    // Show Web Search option

    if (webSearchSection) {

        webSearchSection.style.display =
            "block";

    }

    // Render retrieved sources

    renderSources(
        data.sources || []
    );

} else {

    // Hide Web Search

    if (webSearchSection) {

        webSearchSection.style.display =
            "none";

    }

    // Hide Retrieved Sources content

    sourcesElement.innerHTML = "";

}


            // ==================================================
            // Execution Summary
            // ==========================================================

            providerElement.innerHTML =
                data.provider ||
                "-";


            modelElement.innerHTML =
                data.model ||
                "-";


            if (
                data.confidence !== undefined &&
                data.confidence !== null
            ) {

                confidenceElement.innerHTML =
                    Number(
                        data.confidence
                    ).toFixed(4);

            } else {

                confidenceElement.innerHTML =
                    "-";

            }


            if (
                data.total_time_ms !== undefined &&
                data.total_time_ms !== null
            ) {

                timeElement.innerHTML =
                    `${(
                        Number(data.total_time_ms) / 1000
                    ).toFixed(3)} sec`;

            } else {

                timeElement.innerHTML =
                    "-";

            }


            if (
                data.total_tokens !== undefined &&
                data.total_tokens !== null
            ) {

                tokensElement.innerHTML =
                    data.total_tokens;

            } else {

                tokensElement.innerHTML =
                    "-";

            }


            if (
                data.cost !== undefined &&
                data.cost !== null
            ) {

                costElement.innerHTML =
                    `$${Number(
                        data.cost
                    ).toFixed(6)}`;

            } else {

                costElement.innerHTML =
                    "-";

            }


        } catch (error) {

            console.error(
                "Chat error:",
                error
            );


            answerElement.innerHTML =
                `<span class="text-danger">
                    ${escapeHtml(
                        error.message
                    )}
                </span>`;


            sourcesElement.innerHTML =
                "<p class='text-muted'>No sources available.</p>";

        } finally {

            askButton.disabled = false;

            askButton.innerHTML =
                '<i class="bi bi-stars"></i> Ask AI';


            if (loadingElement) {

                loadingElement.style.display =
                    "none";

            }

        }

    }


    // ==========================================================
// Web Search / Refine Response
// ==========================================================
// ==========================================================
// Web Search / Refine Response
// ==========================================================

async function refineWithWebSearch() {

    const currentAnswer =
        answerElement.innerText.trim();


    if (
        !currentAnswer ||
        currentAnswer ===
            "Your answer will appear here..." ||
        currentAnswer ===
            "Generating answer..."
    ) {

        return;
    }


    // ------------------------------------------------------
    // Show Web Search loading UI
    // Hide only the button/prompt
    // ------------------------------------------------------

    const webSearchPrompt =
        document.getElementById(
            "webSearchPrompt"
        );


    if (webSearchPrompt) {

        webSearchPrompt.style.display =
            "none";

    }


    if (webSearchLoading) {

        webSearchLoading.style.display =
            "flex";

    }


    try {

        console.log(
            "Web Search simulation started."
        );


        // --------------------------------------------------
        // Keep your existing 7 second simulation
        // --------------------------------------------------

        await new Promise(
            function (resolve) {

                setTimeout(
                    resolve,
                    7000
                );

            }
        );


        // --------------------------------------------------
        // Update ONLY the answer
        // --------------------------------------------------

        answerElement.innerHTML = `

            <div class="mb-3">

                <strong>
                    Refined Response
                </strong>

            </div>

            <div>
                ${escapeHtml(currentAnswer)}
            </div>

            <hr>

            <div class="text-muted small">

                This response was generated using web information.

            </div>

        `;


        console.log(
            "Web Search simulation completed."
        );


    } catch (error) {

        console.error(
            "Web Search refinement failed:",
            error
        );


        answerElement.innerHTML = `
            <span class="text-danger">
                Unable to refine the response.
            </span>
        `;


    } finally {

        // --------------------------------------------------
        // Hide loading UI
        // --------------------------------------------------

        if (webSearchLoading) {

            webSearchLoading.style.display =
                "none";

        }


        // --------------------------------------------------
        // Show Web Search button/prompt again
        // --------------------------------------------------

        if (webSearchPrompt) {

            webSearchPrompt.style.display =
                "flex";

        }

    }

}

    // ==========================================================
    // Sources
    // ==========================================================

    function renderSources(sources) {

        sourcesElement.innerHTML = "";


        if (
            !sources ||
            sources.length === 0
        ) {

            sourcesElement.innerHTML =
                "<p class='text-muted'>No sources found.</p>";

            return;
        }


        sources.forEach(
            function (source, index) {

                const div =
                    document.createElement(
                        "div"
                    );


                div.className =
                    "card mb-2";


                div.innerHTML = `

                    <div class="card-body">

                        <strong>
                            Document ${index + 1}
                        </strong>

                        <hr>

                        <pre class="small mb-0">${escapeHtml(
                            JSON.stringify(
                                source,
                                null,
                                2
                            )
                        )}</pre>

                    </div>

                `;


                sourcesElement.appendChild(
                    div
                );

            }
        );

    }


    // ==========================================================
    // Copy Answer
    // ==========================================================

    window.copyAnswer = function () {

        navigator.clipboard.writeText(
            answerElement.innerText
        );

    };


    // ==========================================================
    // HTML Escape
    // ==========================================================

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


});