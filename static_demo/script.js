document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatHistory = document.getElementById('chatHistory');
    const emptyState = document.getElementById('emptyState');
    const sendBtn = document.getElementById('sendBtn');

    // Hardcoded responses to simulate the RAG model
    const mockResponses = [
        {
            keywords: ['article 14', 'equality'],
            base: "Article 14 is a part of the Indian Constitution, but without further context, I can tell you it broadly deals with equality. It states that the State shall not deny to any person equality before the law within the territory of India. However, there are many caveats and legal interpretations of this in various contexts.",
            rag: "Article 14 of the Constitution guarantees equality before law and equal protection of the laws to all persons within the territory of India. It prohibits unreasonable discrimination by the State.",
            source: "constitution.txt"
        },
        {
            keywords: ['fundamental duties', 'duty'],
            base: "Fundamental duties are moral obligations of citizens to help promote a spirit of patriotism and uphold the unity of India. These were added by the 42nd Amendment in 1976. Some of them include respecting the national flag and anthem, protecting the natural environment, and developing scientific temper.",
            rag: "The Fundamental Duties mentioned in the Constitution include abiding by the Constitution, respecting its ideals and institutions, the National Flag and the National Anthem, and to uphold and protect the sovereignty, unity, and integrity of India (Article 51A).",
            source: "constitution.txt"
        },
        {
            keywords: ['president', 'age', 'qualifications'],
            base: "To be the President of India, you must be a citizen of India and have completed a certain age, usually 35 years. You must also be qualified for election as a member of the House of the People.",
            rag: "To be eligible for election as President of India, a person must have completed thirty-five years of age (Article 58).",
            source: "constitution.txt"
        }
    ];

    const defaultResponse = {
        base: "I am an AI, and I can try to answer your question based on my general pre-training data. However, I might not be fully accurate regarding specific constitutional articles without access to the actual documents.",
        rag: "I could not find specific information answering this question in the provided constitutional documents.",
        source: "none"
    };

    function findResponse(query) {
        const lowerQuery = query.toLowerCase();
        for (let r of mockResponses) {
            if (r.keywords.some(kw => lowerQuery.includes(kw))) {
                return r;
            }
        }
        return defaultResponse;
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
