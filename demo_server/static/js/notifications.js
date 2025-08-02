/**
 * Notifications component for handling batch processing notifications
 * and displaying batch results.
 */

// Global notification state
let notificationState = {
    notifications: [],
    unreadCount: 0,
    isLoading: false,
    isOpen: false
};

// Initialize notifications when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Create notification bell if it doesn't exist
    createNotificationBell();
    
    // Fetch notifications initially
    fetchNotifications();
    
    // Set up polling for new notifications (every 30 seconds)
    setInterval(fetchNotifications, 30000);
});

/**
 * Create the notification bell UI element
 */
function createNotificationBell() {
    // Check if notification bell already exists
    if (document.getElementById('notification-bell')) {
        return;
    }
    
    // Create notification bell container
    const bellContainer = document.createElement('div');
    bellContainer.id = 'notification-bell-container';
    bellContainer.className = 'notification-bell-container';
    
    // Create notification bell
    const bell = document.createElement('div');
    bell.id = 'notification-bell';
    bell.className = 'notification-bell';
    bell.innerHTML = '<i class="fas fa-bell"></i>';
    bell.addEventListener('click', toggleNotificationPanel);
    
    // Create notification count badge
    const badge = document.createElement('div');
    badge.id = 'notification-badge';
    badge.className = 'notification-badge hidden';
    badge.textContent = '0';
    
    // Create notification panel
    const panel = document.createElement('div');
    panel.id = 'notification-panel';
    panel.className = 'notification-panel hidden';
    
    // Create notification panel header
    const header = document.createElement('div');
    header.className = 'notification-header';
    header.innerHTML = '<h3>Notifications</h3>';
    
    // Create mark all as read button
    const markAllRead = document.createElement('button');
    markAllRead.className = 'mark-all-read';
    markAllRead.textContent = 'Mark all as read';
    markAllRead.addEventListener('click', markAllNotificationsAsRead);
    header.appendChild(markAllRead);
    
    // Create notification list
    const list = document.createElement('div');
    list.id = 'notification-list';
    list.className = 'notification-list';
    
    // Create notification empty state
    const emptyState = document.createElement('div');
    emptyState.id = 'notification-empty';
    emptyState.className = 'notification-empty';
    emptyState.textContent = 'No notifications';
    
    // Assemble the notification panel
    panel.appendChild(header);
    panel.appendChild(list);
    panel.appendChild(emptyState);
    
    // Add elements to the bell container
    bellContainer.appendChild(bell);
    bellContainer.appendChild(badge);
    bellContainer.appendChild(panel);
    
    // Add the bell container to the page
    // Try to find the header or nav element to append to
    const header_el = document.querySelector('header') || document.querySelector('nav');
    
    if (header_el) {
        header_el.appendChild(bellContainer);
    } else {
        // If no header or nav, add to body
        document.body.insertBefore(bellContainer, document.body.firstChild);
    }
    
    // Add notification styles
    addNotificationStyles();
}

/**
 * Add notification styles to the page
 */
function addNotificationStyles() {
    // Check if styles already exist
    if (document.getElementById('notification-styles')) {
        return;
    }
    
    const styles = document.createElement('style');
    styles.id = 'notification-styles';
    styles.textContent = `
        .notification-bell-container {
            position: relative;
            display: inline-block;
            margin-left: 15px;
        }
        
        .notification-bell {
            cursor: pointer;
            font-size: 1.2rem;
            color: #555;
            padding: 5px;
        }
        
        .notification-bell:hover {
            color: #000;
        }
        
        .notification-badge {
            position: absolute;
            top: -5px;
            right: -5px;
            background-color: #f44336;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .notification-badge.hidden {
            display: none;
        }
        
        .notification-panel {
            position: absolute;
            top: 40px;
            right: 0;
            width: 300px;
            max-height: 400px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .notification-panel.hidden {
            display: none;
        }
        
        .notification-header {
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .notification-header h3 {
            margin: 0;
            font-size: 16px;
        }
        
        .mark-all-read {
            background: none;
            border: none;
            color: #2196F3;
            cursor: pointer;
            font-size: 12px;
            padding: 0;
        }
        
        .notification-list {
            overflow-y: auto;
            max-height: 350px;
            padding: 0;
        }
        
        .notification-item {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .notification-item:hover {
            background-color: #f9f9f9;
        }
        
        .notification-item.unread {
            background-color: #e3f2fd;
        }
        
        .notification-item.unread:hover {
            background-color: #bbdefb;
        }
        
        .notification-title {
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 14px;
        }
        
        .notification-message {
            color: #666;
            font-size: 13px;
            margin-bottom: 5px;
        }
        
        .notification-time {
            color: #999;
            font-size: 11px;
            text-align: right;
        }
        
        .notification-empty {
            padding: 20px;
            text-align: center;
            color: #999;
            font-style: italic;
        }
    `;
    
    document.head.appendChild(styles);
}

/**
 * Toggle the notification panel
 */
function toggleNotificationPanel() {
    const panel = document.getElementById('notification-panel');
    
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
        notificationState.isOpen = true;
        
        // Mark notifications as read when panel is opened
        if (notificationState.unreadCount > 0) {
            const unreadIds = notificationState.notifications
                .filter(n => !n.read)
                .map(n => n.id);
                
            if (unreadIds.length > 0) {
                markNotificationsAsRead(unreadIds);
            }
        }
    } else {
        panel.classList.add('hidden');
        notificationState.isOpen = false;
    }
}

/**
 * Fetch notifications from the server
 */
function fetchNotifications() {
    if (notificationState.isLoading) {
        return;
    }
    
    notificationState.isLoading = true;
    
    fetch('/api/notifications')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch notifications');
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                updateNotifications(data.notifications || []);
            }
        })
        .catch(error => {
            console.error('Error fetching notifications:', error);
        })
        .finally(() => {
            notificationState.isLoading = false;
        });
}

/**
 * Update notifications in the UI
 */
function updateNotifications(notifications) {
    notificationState.notifications = notifications;
    
    // Count unread notifications
    const unreadCount = notifications.filter(n => !n.read).length;
    notificationState.unreadCount = unreadCount;
    
    // Update badge
    const badge = document.getElementById('notification-badge');
    if (unreadCount > 0) {
        badge.textContent = unreadCount > 99 ? '99+' : unreadCount;
        badge.classList.remove('hidden');
    } else {
        badge.classList.add('hidden');
    }
    
    // Update notification list
    const list = document.getElementById('notification-list');
    const emptyState = document.getElementById('notification-empty');
    
    // Clear current list
    list.innerHTML = '';
    
    if (notifications.length === 0) {
        list.classList.add('hidden');
        emptyState.classList.remove('hidden');
    } else {
        list.classList.remove('hidden');
        emptyState.classList.add('hidden');
        
        // Add notifications to list
        notifications.forEach(notification => {
            const item = createNotificationItem(notification);
            list.appendChild(item);
        });
    }
}

/**
 * Create a notification item element
 */
function createNotificationItem(notification) {
    const item = document.createElement('div');
    item.className = `notification-item ${notification.read ? '' : 'unread'}`;
    item.dataset.id = notification.id;
    
    // Create notification title
    const title = document.createElement('div');
    title.className = 'notification-title';
    title.textContent = notification.title || 'Notification';
    
    // Create notification message
    const message = document.createElement('div');
    message.className = 'notification-message';
    message.textContent = notification.message || '';
    
    // Create notification time
    const time = document.createElement('div');
    time.className = 'notification-time';
    time.textContent = formatNotificationTime(notification.timestamp);
    
    // Add click handler
    item.addEventListener('click', () => {
        handleNotificationClick(notification);
    });
    
    // Assemble the notification item
    item.appendChild(title);
    item.appendChild(message);
    item.appendChild(time);
    
    return item;
}

/**
 * Format notification timestamp
 */
function formatNotificationTime(timestamp) {
    if (!timestamp) {
        return '';
    }
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);
    
    if (diffSec < 60) {
        return 'Just now';
    } else if (diffMin < 60) {
        return `${diffMin} minute${diffMin !== 1 ? 's' : ''} ago`;
    } else if (diffHour < 24) {
        return `${diffHour} hour${diffHour !== 1 ? 's' : ''} ago`;
    } else if (diffDay < 7) {
        return `${diffDay} day${diffDay !== 1 ? 's' : ''} ago`;
    } else {
        return date.toLocaleDateString();
    }
}

/**
 * Handle notification click
 */
function handleNotificationClick(notification) {
    // Mark as read if not already
    if (!notification.read) {
        markNotificationsAsRead([notification.id]);
    }
    
    // Handle different notification types
    if (notification.type === 'batch_complete') {
        // Navigate to batch results page
        if (notification.batch_id) {
            window.location.href = `/batch/results/${notification.batch_id}`;
        }
    } else if (notification.url) {
        // Navigate to URL if provided
        window.location.href = notification.url;
    }
}

/**
 * Mark notifications as read
 */
function markNotificationsAsRead(notificationIds) {
    if (!notificationIds || notificationIds.length === 0) {
        return;
    }
    
    fetch('/api/notifications/mark-read', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            notification_ids: notificationIds
        })
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to mark notifications as read');
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                // Update local notifications
                notificationState.notifications = notificationState.notifications.map(n => {
                    if (notificationIds.includes(n.id)) {
                        return { ...n, read: true };
                    }
                    return n;
                });
                
                // Update UI
                updateNotifications(notificationState.notifications);
            }
        })
        .catch(error => {
            console.error('Error marking notifications as read:', error);
        });
}

/**
 * Mark all notifications as read
 */
function markAllNotificationsAsRead() {
    const unreadIds = notificationState.notifications
        .filter(n => !n.read)
        .map(n => n.id);
        
    if (unreadIds.length > 0) {
        markNotificationsAsRead(unreadIds);
    }
}

/**
 * Add a notification programmatically (for testing)
 */
function addTestNotification() {
    const notification = {
        id: 'test-' + Date.now(),
        title: 'Test Notification',
        message: 'This is a test notification',
        timestamp: new Date().toISOString(),
        read: false,
        type: 'test'
    };
    
    notificationState.notifications.unshift(notification);
    updateNotifications(notificationState.notifications);
}
