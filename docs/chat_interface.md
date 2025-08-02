# Medical Report Assistant - Chat Interface Documentation

## Overview

The chat interface component of the Medical Report Assistant provides an interactive, conversational way to interact with the AI assistant for processing medical reports. It offers a modern, user-friendly experience with real-time responses and structured display of medical report information.

## Architecture

The chat interface follows a client-server architecture:

- **Client**: HTML, CSS, and JavaScript for the user interface
- **Server**: Flask backend that processes requests and communicates with the MCP system
- **Communication**: AJAX requests for asynchronous communication

## Key Components

### 1. Chat UI (`chat.html`)

The main chat interface file contains:
- HTML structure for the chat layout
- CSS styles for visual appearance
- JavaScript for interactive functionality
- AJAX communication with the server

### 2. Message Handling

- User message display
- Assistant message display
- File upload handling
- Structured report display
- Markdown formatting

### 3. UI Components

- Sidebar with navigation and features
- Chat header with information and actions
- Chat messages area
- Input area with text input and file upload
- Report cards for structured display of medical information

## Detailed Features

### Interactive Chat

- Real-time message display
- Typing indicators
- Message timestamps
- Avatar display for the assistant
- Markdown formatting support

### File Upload

- Drag-and-drop file upload
- File preview before sending
- Progress indication during upload
- Support for PDF and image files

### Report Processing

- Natural language commands for processing reports
- Structured display of processing results
- Toggle functionality for detailed information
- Visual indicators for processing status

### Structured Display

The chat interface includes a custom implementation for displaying structured medical report data:

- **Report Cards**: Visually appealing cards for displaying medical report summaries
- **Patient Information**: Clearly formatted patient details
- **Summary Section**: Concise summary of the report
- **Extracted Text**: Collapsible section for detailed extracted text
- **Markdown Support**: Proper formatting of medical terminology and structure

## Implementation Details

### HTML Structure

The chat interface uses a structured HTML layout:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Meta tags, title, and CSS links -->
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <!-- App title and navigation -->
            <div class="sidebar-header">
                <div class="app-title">Medical Report Assistant</div>
            </div>
            
            <!-- New chat button -->
            <div class="new-chat-container">
                <button class="new-chat-btn">
                    <i class="bi bi-plus-circle"></i> New Chat
                </button>
            </div>
            
            <!-- Features section -->
            <div class="features-section">
                <!-- List of features -->
            </div>
        </div>
        
        <!-- Chat area -->
        <div class="chat-area">
            <!-- Chat header -->
            <div class="chat-header">
                <div class="chat-title">Medical Report Chat</div>
                <div class="header-actions">
                    <!-- Action buttons -->
                </div>
            </div>
            
            <!-- Chat messages -->
            <div class="chat-messages" id="chat-messages">
                <!-- Messages will be added here dynamically -->
            </div>
            
            <!-- Chat input -->
            <div class="chat-input">
                <div class="input-container">
                    <!-- Text input and file upload -->
                </div>
            </div>
        </div>
    </div>
    
    <!-- JavaScript libraries and custom scripts -->
</body>
</html>
```

### CSS Styling

The chat interface uses custom CSS for styling:

```css
/* Root variables for consistent theming */
:root {
    --primary-color: #3a86ff;
    --primary-gradient: linear-gradient(135deg, #3a86ff, #0066ff);
    --primary-light: #e6f2ff;
    --secondary-color: #6c757d;
    --accent-color: #ff9e00;
    --success-color: #38b000;
    --danger-color: #ff5a5f;
    --light-color: #f8f9fa;
    --dark-color: #212529;
    --border-color: #dee2e6;
    --text-color: #212529;
    --sidebar-width: 240px;
    --header-height: 60px;
    --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 8px rgba(0,0,0,0.1);
    --shadow-lg: 0 8px 16px rgba(0,0,0,0.1);
    --transition-fast: 0.2s ease;
    --transition-normal: 0.3s ease;
    --transition-slow: 0.5s ease;
}

/* Layout styles */
.app-container {
    display: flex;
    height: 100vh;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: var(--shadow-lg);
}

/* Sidebar styles */
.sidebar {
    width: var(--sidebar-width);
    height: 100%;
    border-right: 1px solid rgba(0, 0, 0, 0.05);
    display: flex;
    flex-direction: column;
    background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    box-shadow: var(--shadow-sm);
    position: relative;
    z-index: 10;
    overflow: hidden;
}

/* Chat area styles */
.chat-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
    background-color: #ffffff;
    position: relative;
}

/* Message styles */
.message {
    margin-bottom: 24px;
    max-width: 85%;
    position: relative;
}

/* Report card styles */
.report-card {
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    overflow: hidden;
    margin-top: 10px;
    border: 1px solid #e6f2ff;
}

.report-header {
    background: var(--primary-gradient);
    color: white;
    padding: 12px 16px;
}

.report-section {
    padding: 12px 16px;
    border-bottom: 1px solid #edf2f7;
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}

.info-card {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 10px 12px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.extract-content {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 12px;
    max-height: 400px;
    overflow-y: auto;
    transition: max-height var(--transition-normal);
}

.extract-content.collapsed {
    max-height: 0;
    padding: 0;
    overflow: hidden;
}
```

### JavaScript Functionality

The chat interface uses JavaScript for interactive functionality:

```javascript
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const fileInput = document.getElementById('file-input');
    const fileDropArea = document.getElementById('file-drop-area');
    
    // Variables to track state
    let selectedFile = null;
    
    // Event listeners
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (message || selectedFile) {
            sendMessage(message);
        }
    });
    
    // File upload handling
    fileInput.addEventListener('change', function(e) {
        handleFileSelect(e);
    });
    
    // Drag and drop functionality
    fileDropArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        fileDropArea.classList.add('drag-over');
    });
    
    fileDropArea.addEventListener('dragleave', function() {
        fileDropArea.classList.remove('drag-over');
    });
    
    fileDropArea.addEventListener('drop', function(e) {
        e.preventDefault();
        fileDropArea.classList.remove('drag-over');
        handleFileDrop(e);
    });
    
    // Functions for message handling
    function sendMessage(content) {
        // Add user message to chat
        addUserMessage(content);
        
        // Clear input
        messageInput.value = '';
        
        // Show typing indicator
        addTypingIndicator();
        
        // Prepare form data for file upload
        const formData = new FormData();
        if (content) {
            formData.append('message', content);
        }
        if (selectedFile) {
            formData.append('file', selectedFile);
        }
        
        // Send message to server
        fetch('/chat/send', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            const typingIndicator = document.querySelector('.typing-indicator');
            if (typingIndicator) {
                chatMessages.removeChild(typingIndicator);
            }
            
            // Add assistant response
            addAssistantMessage(data.response, data.tool_result, data.result_data);
            
            // Reset file upload
            selectedFile = null;
            updateFilePreview();
        })
        .catch(error => {
            console.error('Error:', error);
            // Remove typing indicator
            const typingIndicator = document.querySelector('.typing-indicator');
            if (typingIndicator) {
                chatMessages.removeChild(typingIndicator);
            }
            // Add error message
            addAssistantMessage('Sorry, there was an error processing your request. Please try again.');
        });
    }
    
    // Function to add user message to chat
    function addUserMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        
        // Add file preview if available
        if (selectedFile) {
            const filePreviewDiv = document.createElement('div');
            filePreviewDiv.className = 'file-preview';
            filePreviewDiv.innerHTML = `
                <i class="fas fa-file-pdf"></i>
                <div class="file-info">
                    <div class="file-name">${selectedFile.name}</div>
                </div>
            `;
            messageDiv.appendChild(filePreviewDiv);
        }
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = formatDate();
        messageDiv.appendChild(messageTime);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content || '';
        messageDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to add assistant message to chat
    function addAssistantMessage(content, toolResult = null, resultData = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'assistant-avatar';
        avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
        messageDiv.appendChild(avatarDiv);
        
        const messageContainer = document.createElement('div');
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = formatDate();
        messageContainer.appendChild(messageTime);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Check if content is a JSON string containing medical report data
        try {
            if (content && content.includes('"Patient Name"') && content.includes('"Extract Text"')) {
                // This appears to be a medical report summary in JSON format
                const reportData = JSON.parse(content);
                
                // Create a formatted report card
                const reportCard = document.createElement('div');
                reportCard.className = 'report-card';
                
                // Create header
                const reportHeader = document.createElement('div');
                reportHeader.className = 'report-header';
                reportHeader.innerHTML = `<h5><i class="fas fa-file-medical me-2"></i>Medical Report Summary</h5>`;
                reportCard.appendChild(reportHeader);
                
                // Create patient info section
                const patientSection = document.createElement('div');
                patientSection.className = 'report-section';
                
                const patientHeader = document.createElement('div');
                patientHeader.className = 'section-header';
                patientHeader.innerHTML = `<h6><i class="fas fa-user-circle me-2"></i>Patient Information</h6>`;
                patientSection.appendChild(patientHeader);
                
                // Create patient info grid
                const patientGrid = document.createElement('div');
                patientGrid.className = 'info-grid';
                
                // Add patient details
                const patientDetails = [
                    { label: 'Patient Name', value: reportData['Patient Name'] || 'UNKNOWN' },
                    { label: 'Age', value: reportData['Age'] || 'UNKNOWN' },
                    { label: 'Gender', value: reportData['Gender'] || 'UNKNOWN' },
                    { label: 'Date', value: reportData['Date'] || 'UNKNOWN' }
                ];
                
                patientDetails.forEach(detail => {
                    const infoCard = document.createElement('div');
                    infoCard.className = 'info-card';
                    infoCard.innerHTML = `
                        <div class="info-label">${detail.label}</div>
                        <div class="info-value">${detail.value}</div>
                    `;
                    patientGrid.appendChild(infoCard);
                });
                
                patientSection.appendChild(patientGrid);
                reportCard.appendChild(patientSection);
                
                // Create summary section
                const summarySection = document.createElement('div');
                summarySection.className = 'report-section';
                
                const summaryHeader = document.createElement('div');
                summaryHeader.className = 'section-header';
                summaryHeader.innerHTML = `<h6><i class="fas fa-clipboard-list me-2"></i>Report Summary</h6>`;
                summarySection.appendChild(summaryHeader);
                
                const summaryContent = document.createElement('div');
                summaryContent.className = 'summary-content';
                summaryContent.textContent = reportData['Summary'] || 'No summary available';
                summarySection.appendChild(summaryContent);
                
                reportCard.appendChild(summarySection);
                
                // Add extracted text section with toggle
                const extractSection = document.createElement('div');
                extractSection.className = 'report-section';
                
                const extractHeader = document.createElement('div');
                extractHeader.className = 'section-header clickable';
                extractHeader.innerHTML = `
                    <h6><i class="fas fa-file-medical-alt me-2"></i>Extracted Text</h6>
                    <button class="btn-toggle"><i class="fas fa-chevron-down"></i></button>
                `;
                extractSection.appendChild(extractHeader);
                
                const extractContent = document.createElement('div');
                extractContent.className = 'extract-content collapsed';
                
                if (reportData['Extract Text']) {
                    // Create a pre element for the markdown content
                    const preElement = document.createElement('pre');
                    preElement.className = 'markdown-content';
                    preElement.textContent = reportData['Extract Text'];
                    
                    // Parse markdown if available
                    try {
                        preElement.innerHTML = marked.parse(reportData['Extract Text']);
                    } catch (e) {
                        console.error('Error parsing markdown:', e);
                    }
                    
                    extractContent.appendChild(preElement);
                } else {
                    extractContent.textContent = 'No extracted text available';
                }
                
                extractSection.appendChild(extractContent);
                reportCard.appendChild(extractSection);
                
                // Add toggle functionality
                extractHeader.addEventListener('click', function() {
                    extractContent.classList.toggle('collapsed');
                    const icon = extractHeader.querySelector('.btn-toggle i');
                    if (extractContent.classList.contains('collapsed')) {
                        icon.className = 'fas fa-chevron-down';
                    } else {
                        icon.className = 'fas fa-chevron-up';
                    }
                });
                
                // Add the report card to the content div
                contentDiv.appendChild(reportCard);
            } else {
                // Regular message content
                contentDiv.innerHTML = marked.parse(content || '');
            }
        } catch (e) {
            console.error('Error parsing content:', e);
            // Fallback to regular content
            contentDiv.innerHTML = marked.parse(content || '');
        }
        
        messageContainer.appendChild(contentDiv);
        
        // Add tool result if available
        if (toolResult) {
            const toolResultDiv = document.createElement('div');
            toolResultDiv.className = 'tool-result';
            toolResultDiv.innerHTML = marked.parse(toolResult);
            messageContainer.appendChild(toolResultDiv);
        }
        
        // Add View Result button if a result file is available
        if (resultData && resultData.view_url) {
            const actionDiv = document.createElement('div');
            actionDiv.className = 'message-actions';
            actionDiv.innerHTML = `
                <a href="${resultData.view_url}" class="action-button" target="_blank">
                    <i class="fas fa-eye"></i> View Result
                </a>
            `;
            messageContainer.appendChild(actionDiv);
        }
        
        messageDiv.appendChild(messageContainer);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Helper functions
    function formatDate() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant-message typing-indicator';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'assistant-avatar';
        avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
        typingDiv.appendChild(avatarDiv);
        
        const typingContent = document.createElement('div');
        typingContent.className = 'typing-content';
        typingContent.innerHTML = '<span></span><span></span><span></span>';
        typingDiv.appendChild(typingContent);
        
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function handleFileSelect(event) {
        const files = event.target.files;
        if (files.length > 0) {
            selectedFile = files[0];
            updateFilePreview();
        }
    }
    
    function handleFileDrop(event) {
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            selectedFile = files[0];
            fileInput.files = event.dataTransfer.files;
            updateFilePreview();
        }
    }
    
    function updateFilePreview() {
        const filePreview = document.getElementById('file-preview');
        if (selectedFile) {
            filePreview.innerHTML = `
                <div class="selected-file">
                    <i class="fas fa-file-pdf"></i>
                    <div class="file-info">
                        <div class="file-name">${selectedFile.name}</div>
                        <div class="file-size">${formatFileSize(selectedFile.size)}</div>
                    </div>
                    <button type="button" class="remove-file" id="remove-file">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            filePreview.style.display = 'block';
            
            // Add event listener to remove button
            document.getElementById('remove-file').addEventListener('click', function() {
                selectedFile = null;
                fileInput.value = '';
                updateFilePreview();
            });
        } else {
            filePreview.innerHTML = '';
            filePreview.style.display = 'none';
        }
    }
    
    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' B';
        } else if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else {
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
    }
});
```

## Report Card Implementation

The key enhancement to the chat interface is the structured display of medical report summaries using report cards. This implementation:

1. Detects when a message contains JSON data with medical report information
2. Parses the JSON data into a structured format
3. Creates a visually appealing card layout with sections for:
   - Patient information
   - Report summary
   - Extracted text (collapsible)
4. Adds interactive elements like toggle buttons for expanding/collapsing sections
5. Applies proper formatting to the content, including markdown parsing

### Report Card Structure

```
┌─────────────────────────────────────────┐
│           Medical Report Summary        │ ← Report Header
├─────────────────────────────────────────┤
│ Patient Information                     │ ← Section Header
│ ┌─────────────┐ ┌─────────────┐         │
│ │ Patient Name│ │ Age         │         │ ← Info Grid
│ │ John Doe    │ │ 45          │         │
│ └─────────────┘ └─────────────┘         │
│ ┌─────────────┐ ┌─────────────┐         │
│ │ Gender      │ │ Date        │         │
│ │ Male        │ │ 2025-05-01  │         │
│ └─────────────┘ └─────────────┘         │
├─────────────────────────────────────────┤
│ Report Summary                          │ ← Section Header
│                                         │
│ Patient presented with symptoms of...   │ ← Summary Content
│                                         │
├─────────────────────────────────────────┤
│ Extracted Text            [▼]           │ ← Collapsible Section
│                                         │
│ (Collapsed by default, expands to show  │
│  the full extracted text with markdown  │
│  formatting)                            │
│                                         │
└─────────────────────────────────────────┘
```

### Detection Logic

```javascript
try {
    if (content && content.includes('"Patient Name"') && content.includes('"Extract Text"')) {
        // This appears to be a medical report summary in JSON format
        const reportData = JSON.parse(content);
        
        // Create formatted report card...
    } else {
        // Regular message content
        contentDiv.innerHTML = marked.parse(content || '');
    }
} catch (e) {
    console.error('Error parsing content:', e);
    // Fallback to regular content
    contentDiv.innerHTML = marked.parse(content || '');
}
```

### Toggle Functionality

```javascript
extractHeader.addEventListener('click', function() {
    extractContent.classList.toggle('collapsed');
    const icon = extractHeader.querySelector('.btn-toggle i');
    if (extractContent.classList.contains('collapsed')) {
        icon.className = 'fas fa-chevron-down';
    } else {
        icon.className = 'fas fa-chevron-up';
    }
});
```

## Responsive Design

The chat interface is designed to be responsive and work well on both desktop and mobile devices:

- Fluid layout that adapts to different screen sizes
- Media queries for adjusting components on smaller screens
- Touch-friendly interface elements
- Appropriate font sizes and spacing for readability

### Mobile Optimizations

```css
@media (max-width: 768px) {
    .app-container {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
        height: auto;
        max-height: 60px;
        overflow: hidden;
        transition: max-height var(--transition-normal);
    }
    
    .sidebar.expanded {
        max-height: 100vh;
    }
    
    .chat-area {
        height: calc(100vh - 60px);
    }
    
    .info-grid {
        grid-template-columns: 1fr;
    }
}
```

## Accessibility Considerations

The chat interface includes several accessibility enhancements:

- Semantic HTML structure
- ARIA attributes for screen readers
- Keyboard navigation support
- Sufficient color contrast
- Focus indicators for interactive elements

## Performance Optimizations

- Efficient DOM manipulation
- Event delegation for dynamic elements
- Lazy loading of resources
- Debouncing for input events
- Memory management for large conversations

## Security Considerations

- Input sanitization to prevent XSS attacks
- Content Security Policy implementation
- Secure handling of file uploads
- Error handling and graceful degradation

## Future Enhancements

- Voice input and output support
- Real-time collaboration features
- Advanced file handling capabilities
- Integration with electronic health record systems
- Customizable themes and layouts
- Offline support with service workers
- Multi-language support
- Accessibility improvements
