<?php
/**
 * Plugin Name: RAG Chat Widget
 * Plugin URI: https://github.com/your-username/rag-widget
 * Description: Add an AI-powered chat widget to your WordPress site using RAG (Retrieval-Augmented Generation) technology.
 * Version: 1.0.0
 * Author: Your Name
 * Author URI: https://yoursite.com
 * License: MIT
 * Text Domain: rag-widget
 * Domain Path: /languages
 *
 * @package RAGWidget
 */

// Prevent direct access
if (!defined('ABSPATH')) {
    exit;
}

// Define plugin constants
define('RAG_WIDGET_VERSION', '1.0.0');
define('RAG_WIDGET_PLUGIN_URL', plugin_dir_url(__FILE__));
define('RAG_WIDGET_PLUGIN_PATH', plugin_dir_path(__FILE__));

/**
 * Main RAG Widget class
 */
class RAGWidget {
    
    /**
     * Constructor
     */
    public function __construct() {
        add_action('init', array($this, 'init'));
        add_action('wp_enqueue_scripts', array($this, 'enqueue_scripts'));
        add_action('wp_footer', array($this, 'render_widget'));
        add_action('admin_menu', array($this, 'add_admin_menu'));
        add_action('admin_init', array($this, 'admin_init'));
        
        // Ajax actions
        add_action('wp_ajax_rag_widget_test', array($this, 'test_api_connection'));
        add_action('wp_ajax_nopriv_rag_widget_test', array($this, 'test_api_connection'));
    }
    
    /**
     * Initialize the plugin
     */
    public function init() {
        // Load text domain for translations
        load_plugin_textdomain('rag-widget', false, dirname(plugin_basename(__FILE__)) . '/languages');
    }
    
    /**
     * Enqueue scripts and styles
     */
    public function enqueue_scripts() {
        // Only load on frontend if widget is enabled
        if (!$this->is_widget_enabled()) {
            return;
        }
        
        $api_key = get_option('rag_widget_api_key', '');
        $api_url = get_option('rag_widget_api_url', 'http://localhost:8001');
        
        // Don't load if no API key is set
        if (empty($api_key)) {
            return;
        }
        
        // Enqueue the widget loader script
        wp_enqueue_script(
            'rag-widget-loader',
            RAG_WIDGET_PLUGIN_URL . 'js/widget-loader.js',
            array(),
            RAG_WIDGET_VERSION,
            true
        );
        
        // Pass configuration to JavaScript
        wp_localize_script('rag-widget-loader', 'ragWidgetConfig', array(
            'apiKey' => $api_key,
            'apiUrl' => $api_url,
            'theme' => get_option('rag_widget_theme', 'default'),
            'position' => get_option('rag_widget_position', 'bottom-right'),
            'primaryColor' => get_option('rag_widget_primary_color', '#667eea'),
            'secondaryColor' => get_option('rag_widget_secondary_color', '#764ba2'),
            'title' => get_option('rag_widget_title', 'AI Assistant'),
            'welcomeMessage' => get_option('rag_widget_welcome_message', 'Hello! How can I help you today?'),
            'placeholder' => get_option('rag_widget_placeholder', 'Type your message...'),
            'zIndex' => get_option('rag_widget_z_index', '999999')
        ));
    }
    
    /**
     * Render the widget HTML
     */
    public function render_widget() {
        if (!$this->is_widget_enabled()) {
            return;
        }
        
        $api_key = get_option('rag_widget_api_key', '');
        if (empty($api_key)) {
            return;
        }
        
        // Widget is loaded via JavaScript, no HTML needed here
    }
    
    /**
     * Check if widget should be displayed
     */
    private function is_widget_enabled() {
        $enabled = get_option('rag_widget_enabled', '1');
        $show_on_pages = get_option('rag_widget_show_on_pages', array('all'));
        
        if ($enabled !== '1') {
            return false;
        }
        
        // Check page restrictions
        if (in_array('all', $show_on_pages)) {
            return true;
        }
        
        if (is_front_page() && in_array('home', $show_on_pages)) {
            return true;
        }
        
        if (is_page() && in_array('pages', $show_on_pages)) {
            return true;
        }
        
        if (is_single() && in_array('posts', $show_on_pages)) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Add admin menu
     */
    public function add_admin_menu() {
        add_options_page(
            __('RAG Widget Settings', 'rag-widget'),
            __('RAG Widget', 'rag-widget'),
            'manage_options',
            'rag-widget',
            array($this, 'admin_page')
        );
    }
    
    /**
     * Initialize admin settings
     */
    public function admin_init() {
        register_setting('rag_widget_settings', 'rag_widget_enabled');
        register_setting('rag_widget_settings', 'rag_widget_api_key');
        register_setting('rag_widget_settings', 'rag_widget_api_url');
        register_setting('rag_widget_settings', 'rag_widget_theme');
        register_setting('rag_widget_settings', 'rag_widget_position');
        register_setting('rag_widget_settings', 'rag_widget_primary_color');
        register_setting('rag_widget_settings', 'rag_widget_secondary_color');
        register_setting('rag_widget_settings', 'rag_widget_title');
        register_setting('rag_widget_settings', 'rag_widget_welcome_message');
        register_setting('rag_widget_settings', 'rag_widget_placeholder');
        register_setting('rag_widget_settings', 'rag_widget_z_index');
        register_setting('rag_widget_settings', 'rag_widget_show_on_pages');
    }
    
    /**
     * Admin page content
     */
    public function admin_page() {
        ?>
        <div class="wrap">
            <h1><?php echo esc_html(get_admin_page_title()); ?></h1>
            
            <form action="options.php" method="post">
                <?php
                settings_fields('rag_widget_settings');
                do_settings_sections('rag_widget_settings');
                ?>
                
                <table class="form-table">
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_enabled"><?php _e('Enable Widget', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <input type="checkbox" id="rag_widget_enabled" name="rag_widget_enabled" value="1" 
                                   <?php checked(get_option('rag_widget_enabled', '1'), '1'); ?> />
                            <p class="description"><?php _e('Enable or disable the RAG chat widget.', 'rag-widget'); ?></p>
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_api_key"><?php _e('API Key', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <input type="text" id="rag_widget_api_key" name="rag_widget_api_key" 
                                   value="<?php echo esc_attr(get_option('rag_widget_api_key', '')); ?>" 
                                   class="regular-text" />
                            <p class="description"><?php _e('Your RAG API key for authentication.', 'rag-widget'); ?></p>
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_api_url"><?php _e('API URL', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <input type="url" id="rag_widget_api_url" name="rag_widget_api_url" 
                                   value="<?php echo esc_attr(get_option('rag_widget_api_url', 'http://localhost:8001')); ?>" 
                                   class="regular-text" />
                            <p class="description"><?php _e('The URL of your RAG API server.', 'rag-widget'); ?></p>
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_theme"><?php _e('Theme', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <select id="rag_widget_theme" name="rag_widget_theme">
                                <option value="default" <?php selected(get_option('rag_widget_theme', 'default'), 'default'); ?>>Default</option>
                                <option value="dark" <?php selected(get_option('rag_widget_theme', 'default'), 'dark'); ?>>Dark</option>
                                <option value="blue" <?php selected(get_option('rag_widget_theme', 'default'), 'blue'); ?>>Blue</option>
                                <option value="green" <?php selected(get_option('rag_widget_theme', 'default'), 'green'); ?>>Green</option>
                                <option value="purple" <?php selected(get_option('rag_widget_theme', 'default'), 'purple'); ?>>Purple</option>
                                <option value="orange" <?php selected(get_option('rag_widget_theme', 'default'), 'orange'); ?>>Orange</option>
                                <option value="red" <?php selected(get_option('rag_widget_theme', 'default'), 'red'); ?>>Red</option>
                            </select>
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_position"><?php _e('Position', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <select id="rag_widget_position" name="rag_widget_position">
                                <option value="bottom-right" <?php selected(get_option('rag_widget_position', 'bottom-right'), 'bottom-right'); ?>>Bottom Right</option>
                                <option value="bottom-left" <?php selected(get_option('rag_widget_position', 'bottom-right'), 'bottom-left'); ?>>Bottom Left</option>
                                <option value="top-right" <?php selected(get_option('rag_widget_position', 'bottom-right'), 'top-right'); ?>>Top Right</option>
                                <option value="top-left" <?php selected(get_option('rag_widget_position', 'bottom-right'), 'top-left'); ?>>Top Left</option>
                            </select>
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_title"><?php _e('Widget Title', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <input type="text" id="rag_widget_title" name="rag_widget_title" 
                                   value="<?php echo esc_attr(get_option('rag_widget_title', 'AI Assistant')); ?>" 
                                   class="regular-text" />
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_welcome_message"><?php _e('Welcome Message', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <textarea id="rag_widget_welcome_message" name="rag_widget_welcome_message" 
                                      rows="3" cols="50" class="large-text"><?php echo esc_textarea(get_option('rag_widget_welcome_message', 'Hello! How can I help you today?')); ?></textarea>
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_placeholder"><?php _e('Input Placeholder', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <input type="text" id="rag_widget_placeholder" name="rag_widget_placeholder" 
                                   value="<?php echo esc_attr(get_option('rag_widget_placeholder', 'Type your message...')); ?>" 
                                   class="regular-text" />
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_primary_color"><?php _e('Primary Color', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <input type="color" id="rag_widget_primary_color" name="rag_widget_primary_color" 
                                   value="<?php echo esc_attr(get_option('rag_widget_primary_color', '#667eea')); ?>" />
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_secondary_color"><?php _e('Secondary Color', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <input type="color" id="rag_widget_secondary_color" name="rag_widget_secondary_color" 
                                   value="<?php echo esc_attr(get_option('rag_widget_secondary_color', '#764ba2')); ?>" />
                        </td>
                    </tr>
                    
                    <tr>
                        <th scope="row">
                            <label for="rag_widget_show_on_pages"><?php _e('Show On Pages', 'rag-widget'); ?></label>
                        </th>
                        <td>
                            <?php
                            $show_on_pages = get_option('rag_widget_show_on_pages', array('all'));
                            if (!is_array($show_on_pages)) {
                                $show_on_pages = array('all');
                            }
                            ?>
                            <label>
                                <input type="checkbox" name="rag_widget_show_on_pages[]" value="all" 
                                       <?php checked(in_array('all', $show_on_pages)); ?> />
                                <?php _e('All Pages', 'rag-widget'); ?>
                            </label><br>
                            <label>
                                <input type="checkbox" name="rag_widget_show_on_pages[]" value="home" 
                                       <?php checked(in_array('home', $show_on_pages)); ?> />
                                <?php _e('Home Page', 'rag-widget'); ?>
                            </label><br>
                            <label>
                                <input type="checkbox" name="rag_widget_show_on_pages[]" value="pages" 
                                       <?php checked(in_array('pages', $show_on_pages)); ?> />
                                <?php _e('Pages', 'rag-widget'); ?>
                            </label><br>
                            <label>
                                <input type="checkbox" name="rag_widget_show_on_pages[]" value="posts" 
                                       <?php checked(in_array('posts', $show_on_pages)); ?> />
                                <?php _e('Posts', 'rag-widget'); ?>
                            </label>
                        </td>
                    </tr>
                </table>
                
                <?php submit_button(); ?>
                
                <h2><?php _e('API Connection Test', 'rag-widget'); ?></h2>
                <p><?php _e('Click the button below to test your API connection:', 'rag-widget'); ?></p>
                <button type="button" id="test-api-connection" class="button"><?php _e('Test Connection', 'rag-widget'); ?></button>
                <div id="test-result"></div>
            </form>
        </div>
        
        <script>
        jQuery(document).ready(function($) {
            $('#test-api-connection').click(function() {
                var button = $(this);
                var result = $('#test-result');
                
                button.prop('disabled', true).text('<?php _e('Testing...', 'rag-widget'); ?>');
                result.html('<p><?php _e('Testing API connection...', 'rag-widget'); ?></p>');
                
                $.ajax({
                    url: ajaxurl,
                    type: 'POST',
                    data: {
                        action: 'rag_widget_test',
                        api_key: $('#rag_widget_api_key').val(),
                        api_url: $('#rag_widget_api_url').val(),
                        nonce: '<?php echo wp_create_nonce('rag_widget_test'); ?>'
                    },
                    success: function(response) {
                        if (response.success) {
                            result.html('<div class="notice notice-success"><p>' + response.data + '</p></div>');
                        } else {
                            result.html('<div class="notice notice-error"><p>' + response.data + '</p></div>');
                        }
                    },
                    error: function() {
                        result.html('<div class="notice notice-error"><p><?php _e('Connection test failed.', 'rag-widget'); ?></p></div>');
                    },
                    complete: function() {
                        button.prop('disabled', false).text('<?php _e('Test Connection', 'rag-widget'); ?>');
                    }
                });
            });
        });
        </script>
        <?php
    }
    
    /**
     * Test API connection
     */
    public function test_api_connection() {
        if (!wp_verify_nonce($_POST['nonce'], 'rag_widget_test')) {
            wp_die('Security check failed');
        }
        
        $api_key = sanitize_text_field($_POST['api_key']);
        $api_url = esc_url_raw($_POST['api_url']);
        
        if (empty($api_key) || empty($api_url)) {
            wp_send_json_error(__('API key and URL are required.', 'rag-widget'));
        }
        
        // Test the connection
        $response = wp_remote_get($api_url . '/api/status', array(
            'headers' => array(
                'Authorization' => 'Bearer ' . $api_key
            ),
            'timeout' => 10
        ));
        
        if (is_wp_error($response)) {
            wp_send_json_error(__('Connection failed: ', 'rag-widget') . $response->get_error_message());
        }
        
        $status_code = wp_remote_retrieve_response_code($response);
        $body = wp_remote_retrieve_body($response);
        
        if ($status_code === 200) {
            $data = json_decode($body, true);
            if ($data && isset($data['status']) && $data['status'] === 'healthy') {
                wp_send_json_success(__('API connection successful!', 'rag-widget'));
            } else {
                wp_send_json_error(__('API responded but status is not healthy.', 'rag-widget'));
            }
        } else {
            wp_send_json_error(__('API connection failed with status code: ', 'rag-widget') . $status_code);
        }
    }
}

// Initialize the plugin
new RAGWidget();

/**
 * Activation hook
 */
register_activation_hook(__FILE__, function() {
    // Set default options
    add_option('rag_widget_enabled', '1');
    add_option('rag_widget_theme', 'default');
    add_option('rag_widget_position', 'bottom-right');
    add_option('rag_widget_title', 'AI Assistant');
    add_option('rag_widget_welcome_message', 'Hello! How can I help you today?');
    add_option('rag_widget_placeholder', 'Type your message...');
    add_option('rag_widget_primary_color', '#667eea');
    add_option('rag_widget_secondary_color', '#764ba2');
    add_option('rag_widget_z_index', '999999');
    add_option('rag_widget_show_on_pages', array('all'));
});

/**
 * Deactivation hook
 */
register_deactivation_hook(__FILE__, function() {
    // Clean up if needed
});

/**
 * Uninstall hook
 */
register_uninstall_hook(__FILE__, function() {
    // Remove options
    delete_option('rag_widget_enabled');
    delete_option('rag_widget_api_key');
    delete_option('rag_widget_api_url');
    delete_option('rag_widget_theme');
    delete_option('rag_widget_position');
    delete_option('rag_widget_title');
    delete_option('rag_widget_welcome_message');
    delete_option('rag_widget_placeholder');
    delete_option('rag_widget_primary_color');
    delete_option('rag_widget_secondary_color');
    delete_option('rag_widget_z_index');
    delete_option('rag_widget_show_on_pages');
});
?>