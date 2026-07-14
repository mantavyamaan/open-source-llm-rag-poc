document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatHistory = document.getElementById('chatHistory');
    const emptyState = document.getElementById('emptyState');
    const sendBtn = document.getElementById('sendBtn');

    // File upload elements
    const uploadBox = document.getElementById('uploadBox');
    const fileUpload = document.getElementById('fileUpload');
    const uploadSuccess = document.getElementById('uploadSuccess');
    const fileList = document.getElementById('fileList');

    const demoResponse = {
        base: "This is a static portfolio demo website designed to showcase the UI. The actual backend AI model is disconnected to save on cloud GPU costs.",
        rag: "To experience the fully functional DocuMind AI, please visit the project's GitHub repository. Follow the installation steps outlined in the README.md file to run the open-source model locally on your own machine!",
        source: "github.com/mantavyamaan/DocuMind-AI"
    };

    // File upload simulation logic
    if (uploadBox && fileUpload) {
        uploadBox.addEventListener('click', () => {
            fileUpload.click();
        });

        fileUpload.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                const fileName = e.target.files[0].name;
                
                // Show success message
                uploadSuccess.innerHTML = `✅ Saved <code>${fileName}</code> to data/ folder! (Mock Demo)`;
                uploadSuccess.style.display = 'block';

                // Add to file list if not already there
                const existingFiles = Array.from(fileList.querySelectorAll('li')).map(li => li.innerText.replace('📄', '').trim());
                if (!existingFiles.includes(fileName)) {
                    const li = document.createElement('li');
                    li.innerHTML = `<span class="file-icon">📄</span> ${fileName}`;
                    fileList.appendChild(li);
                }

                // Reset input so the same file can be selected again if needed
                fileUpload.value = '';
            }
        });
    }

    function findResponse(query) {
        return demoResponse;
    }

    function appendUserMessage(text) {
        if (emptyState) emptyState.style.display = 'none';
        
        const div = document.createElement('div');
        div.className = 'message user';
        div.innerHTML = `
            <div class="msg-label">👤 You</div>
            <div class="msg-content">${escapeHTML(text)}</div>
        `;
        chatHistory.appendChild(div);
        scrollToBottom();
    }

    function appendAssistantSkeleton() {
        const div = document.createElement('div');
        div.className = 'message assistant loading-msg';
        div.innerHTML = `
            <div class="msg-label">🤖 Constitution AI (Generating...)</div>
            <div class="msg-content" style="padding: 10px 20px;">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatHistory.appendChild(div);
        scrollToBottom();
        return div;
    }

    function simulateTyping(element, text, speed = 20, callback) {
        let i = 0;
        element.innerHTML = '';
        
        function typeWriter() {
            if (i < text.length) {
                element.innerHTML += text.charAt(i);
                i++;
                scrollToBottom();
                setTimeout(typeWriter, speed);
            } else if (callback) {
                callback();
            }
        }
        typeWriter();
    }

    function replaceSkeletonWithResponse(skeletonElement, responseObj) {
        skeletonElement.classList.remove('loading-msg');
        skeletonElement.innerHTML = `
            <div class="msg-label">🤖 Constitution AI</div>
            <div class="split-response">
                <div class="model-col">
                    <h4>Base Open-source LLM</h4>
                    <div class="content-base"></div>
                </div>
                <div class="model-col" style="border-color: rgba(0, 204, 150, 0.4); box-shadow: 0 4px 20px rgba(0, 204, 150, 0.05);">
                    <h4 style="color: #00cc96;">RAG-Optimized LLM</h4>
                    <div class="content-rag"></div>
                    ${responseObj.source !== 'none' ? `<div class="source-box"><strong>Source:</strong> ${responseObj.source}</div>` : ''}
                </div>
            </div>
        `;

        const baseContainer = skeletonElement.querySelector('.content-base');
        const ragContainer = skeletonElement.querySelector('.content-rag');
        const sourceBox = skeletonElement.querySelector('.source-box');
        
        if (sourceBox) sourceBox.style.opacity = '0';

        // Animate typing for Base
        simulateTyping(baseContainer, responseObj.base, 10, () => {
            // Then animate typing for RAG
            simulateTyping(ragContainer, responseObj.rag, 15, () => {
                if(sourceBox) {
                    sourceBox.style.transition = 'opacity 0.5s ease';
                    sourceBox.style.opacity = '1';
                }
            });
        });
    }

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        // Disable input
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;

        // Add user msg
        appendUserMessage(text);
        
        // Find answer
        const answer = findResponse(text);

        // Add skeleton
        const skeleton = appendAssistantSkeleton();

        // Simulate network delay
        setTimeout(() => {
            replaceSkeletonWithResponse(skeleton, answer);
            
            // Re-enable after a short delay mimicking streaming finish
            setTimeout(() => {
                userInput.disabled = false;
                sendBtn.disabled = false;
                userInput.focus();
            }, 3000);
            
        }, 800);
    });

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag])
        );
    }
});
