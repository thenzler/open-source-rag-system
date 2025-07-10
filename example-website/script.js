// TechCorp Solutions Demo Website JavaScript

class TechCorpDemo {
    constructor() {
        this.initializeEventListeners();
        this.initializeAnimations();
        this.initializeContactForm();
    }

    initializeEventListeners() {
        // CTA Button click handler
        const ctaButton = document.querySelector('.cta-button');
        if (ctaButton) {
            ctaButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.scrollToSection('.features');
            });
        }

        // Support link handlers
        const supportLinks = document.querySelectorAll('.support-link');
        supportLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleSupportLinkClick(link);
            });
        });

        // Mobile menu toggle (for future enhancement)
        this.setupMobileMenu();
    }

    initializeAnimations() {
        // Intersection Observer for fade-in animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, {
            threshold: 0.1
        });

        // Observe all sections
        const sections = document.querySelectorAll('section');
        sections.forEach(section => {
            observer.observe(section);
        });
    }

    initializeContactForm() {
        const contactForm = document.getElementById('contactForm');
        if (contactForm) {
            contactForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleContactFormSubmit(e);
            });
        }
    }

    handleContactFormSubmit(e) {
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData);
        
        // Validate form
        if (!this.validateContactForm(data)) {
            return;
        }

        // Simulate form submission
        this.showNotification('Thank you for your message! We\'ll get back to you soon.', 'success');
        
        // Reset form
        e.target.reset();
    }

    validateContactForm(data) {
        const required = ['name', 'email', 'subject', 'message'];
        for (let field of required) {
            if (!data[field] || data[field].trim() === '') {
                this.showNotification(`Please fill in the ${field} field.`, 'error');
                return false;
            }
        }

        // Basic email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(data.email)) {
            this.showNotification('Please enter a valid email address.', 'error');
            return false;
        }

        return true;
    }

    handleSupportLinkClick(link) {
        const linkText = link.textContent.toLowerCase();
        
        switch(linkText) {
            case 'view docs':
                this.showNotification('Documentation would open here in a real implementation.', 'info');
                break;
            case 'start chat':
                this.showNotification('This would open the chat widget. Try the widget in the bottom corner!', 'info');
                break;
            case 'schedule training':
                this.showNotification('Training scheduler would open here.', 'info');
                break;
            case 'contact support':
                this.scrollToSection('.contact-form');
                break;
            default:
                this.showNotification('This feature would be available in the full version.', 'info');
        }
    }

    scrollToSection(selector) {
        const element = document.querySelector(selector);
        if (element) {
            element.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;

        // Add styles
        this.addNotificationStyles();

        // Add to DOM
        document.body.appendChild(notification);

        // Close button handler
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            this.removeNotification(notification);
        });

        // Auto-remove after 5 seconds
        setTimeout(() => {
            this.removeNotification(notification);
        }, 5000);

        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
    }

    removeNotification(notification) {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    addNotificationStyles() {
        // Check if styles already exist
        if (document.querySelector('#notification-styles')) {
            return;
        }

        const styles = document.createElement('style');
        styles.id = 'notification-styles';
        styles.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                max-width: 400px;
                z-index: 10000;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                transform: translateX(450px);
                transition: transform 0.3s ease;
            }

            .notification.show {
                transform: translateX(0);
            }

            .notification-content {
                padding: 16px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .notification-message {
                flex: 1;
                font-weight: 500;
                line-height: 1.4;
            }

            .notification-close {
                background: none;
                border: none;
                font-size: 20px;
                cursor: pointer;
                margin-left: 10px;
                opacity: 0.7;
                transition: opacity 0.3s ease;
            }

            .notification-close:hover {
                opacity: 1;
            }

            .notification-success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }

            .notification-error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }

            .notification-info {
                background: #e6f3ff;
                color: #004085;
                border: 1px solid #b8daff;
            }

            @media (max-width: 768px) {
                .notification {
                    left: 20px;
                    right: 20px;
                    max-width: none;
                    transform: translateY(-100px);
                }

                .notification.show {
                    transform: translateY(0);
                }
            }
        `;
        document.head.appendChild(styles);
    }

    setupMobileMenu() {
        // Mobile menu toggle (placeholder for future enhancement)
        const navMenu = document.querySelector('.nav-menu');
        if (window.innerWidth <= 768) {
            // Mobile menu logic would go here
            console.log('Mobile menu setup - would be implemented here');
        }
    }
}

// RAG Widget Integration Helper
class RAGIntegration {
    constructor() {
        this.initializeWidgetHelpers();
    }

    initializeWidgetHelpers() {
        // Add widget status indicator
        this.addWidgetStatusIndicator();
        
        // Add widget interaction helpers
        this.addWidgetInteractionHelpers();
    }

    addWidgetStatusIndicator() {
        // Create a small indicator showing widget status
        const indicator = document.createElement('div');
        indicator.id = 'widget-status';
        indicator.innerHTML = `
            <div class="widget-status-content">
                <span class="widget-status-text">RAG Widget Demo Active</span>
                <span class="widget-status-dot"></span>
            </div>
        `;

        // Add styles
        const styles = document.createElement('style');
        styles.textContent = `
            #widget-status {
                position: fixed;
                top: 90px;
                right: 20px;
                background: rgba(102, 126, 234, 0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 20px;
                font-size: 12px;
                z-index: 999;
                backdrop-filter: blur(10px);
            }

            .widget-status-content {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .widget-status-dot {
                width: 6px;
                height: 6px;
                background: #4ade80;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            @media (max-width: 768px) {
                #widget-status {
                    top: 70px;
                    right: 10px;
                    font-size: 10px;
                    padding: 6px 10px;
                }
            }
        `;
        document.head.appendChild(styles);
        document.body.appendChild(indicator);

        // Remove indicator after 10 seconds
        setTimeout(() => {
            indicator.style.opacity = '0';
            setTimeout(() => {
                if (indicator.parentNode) {
                    indicator.parentNode.removeChild(indicator);
                }
            }, 300);
        }, 10000);
    }

    addWidgetInteractionHelpers() {
        // Add suggested questions based on current page
        const currentPage = window.location.pathname;
        const suggestions = this.getPageSuggestions(currentPage);
        
        if (suggestions.length > 0) {
            this.addSuggestedQuestions(suggestions);
        }
    }

    getPageSuggestions(pathname) {
        const suggestions = {
            '/index.html': [
                'How do I integrate the RAG API?',
                'What are the system requirements?',
                'Show me a code example',
                'How fast are the responses?'
            ],
            '/about.html': [
                'Tell me about your company',
                'Who is on your team?',
                'What are your company values?',
                'How many companies use your service?'
            ],
            '/contact.html': [
                'How do I get support?',
                'What are your business hours?',
                'Do you offer enterprise pricing?',
                'How do I schedule training?'
            ]
        };

        return suggestions[pathname] || suggestions['/index.html'];
    }

    addSuggestedQuestions(suggestions) {
        // This would integrate with the widget to show suggested questions
        // For demo purposes, we'll just log them
        console.log('Suggested questions for this page:', suggestions);
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TechCorpDemo();
    new RAGIntegration();
});

// Export for potential use by other scripts
window.TechCorpDemo = TechCorpDemo;
window.RAGIntegration = RAGIntegration;