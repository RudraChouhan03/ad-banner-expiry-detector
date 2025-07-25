/**
 * Ad Banner Expiry Detection System - Frontend JavaScript
 * Handles UI interactions and file upload logic
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // File input handling
    const fileInput = document.getElementById('file');  // Get file input element
    const fileNameDisplay = document.getElementById('file-name');  // Get file name display element
    const fileDropArea = document.querySelector('.file-input-container');  // Get file drop area
    
    if (fileInput && fileNameDisplay) {
        // Update displayed filename when file is selected
        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files.length > 0) {
                const file = this.files[0];  // Get selected file
                fileNameDisplay.textContent = file.name;  // Display filename
                
                // Validate file size and type
                validateFile(file);  // Validate the file
            } else {
                fileNameDisplay.textContent = 'No file selected';  // Reset display
            }
        });
    }
    
    // Add drag and drop functionality if drop area exists
    if (fileDropArea) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, preventDefaults, false);  // Prevent default behavior
            document.body.addEventListener(eventName, preventDefaults, false);  // Prevent default behavior
        });
        
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, highlight, false);  // Highlight drop area
        });
        
        // Remove highlight when item is dragged out or dropped
        ['dragleave', 'drop'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, unhighlight, false);  // Remove highlight
        });
        
        // Handle dropped files
        fileDropArea.addEventListener('drop', handleDrop, false);  // Handle file drop
    }
    
    // Initialize tooltips
    const tooltips = document.querySelectorAll('.tooltip');  // Get all tooltips
    if (tooltips.length > 0) {
        tooltips.forEach(tooltip => {
            // Mobile tooltip handling (tap to show/hide)
            tooltip.addEventListener('click', function(e) {
                e.stopPropagation();  // Prevent event bubbling
                
                // Toggle tooltip visibility
                const tooltipText = this.querySelector('.tooltip-text');  // Get tooltip text element
                if (tooltipText) {
                    tooltipText.style.visibility = tooltipText.style.visibility === 'visible' ? 'hidden' : 'visible';  // Toggle visibility
                    tooltipText.style.opacity = tooltipText.style.opacity === '1' ? '0' : '1';  // Toggle opacity
                }
            });
        });
        
        // Hide tooltips when clicking elsewhere
        document.addEventListener('click', function() {
            tooltips.forEach(tooltip => {
                const tooltipText = tooltip.querySelector('.tooltip-text');  // Get tooltip text element
                if (tooltipText) {
                    tooltipText.style.visibility = 'hidden';  // Hide tooltip
                    tooltipText.style.opacity = '0';  // Set opacity to 0
                }
            });
        });
    }
    
    // Helper functions
    function preventDefaults(e) {
        e.preventDefault();  // Prevent default behavior
        e.stopPropagation();  // Stop propagation
    }
    
    function highlight() {
        fileDropArea.classList.add('highlight');  // Add highlight class
    }
    
    function unhighlight() {
        fileDropArea.classList.remove('highlight');  // Remove highlight class
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;  // Get data transfer object
        const files = dt.files;  // Get dropped files
        
        if (files && files.length > 0) {
            fileInput.files = files;  // Set files to input
            fileNameDisplay.textContent = files[0].name;  // Display filename
            
            // Validate file
            validateFile(files[0]);  // Validate the file
        }
    }
    
    function validateFile(file) {
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];  // Allowed file types
        const maxSize = 16 * 1024 * 1024;  // 16MB max size
        
        // Check file type
        if (!allowedTypes.includes(file.type)) {
            showError('Invalid file type. Please upload a JPG or PNG image.');  // Show error
            fileInput.value = '';  // Clear input
            fileNameDisplay.textContent = 'No file selected';  // Reset display
            return false;  // Return false
        }
        
        // Check file size
        if (file.size > maxSize) {
            showError('File is too large. Maximum size is 16MB.');  // Show error
            fileInput.value = '';  // Clear input
            fileNameDisplay.textContent = 'No file selected';  // Reset display
            return false;  // Return false
        }
        
        return true;  // Return true if valid
    }
    
    function showError(message) {
        // Create error element if it doesn't exist
        let errorContainer = document.querySelector('.messages');  // Get message container
        
        if (!errorContainer) {
            errorContainer = document.createElement('div');  // Create container
            errorContainer.className = 'messages';  // Set class
            const form = document.querySelector('.upload-form');  // Get form
            form.parentNode.insertBefore(errorContainer, form);  // Insert before form
        }
        
        // Create error message
        const errorElement = document.createElement('div');  // Create error element
        errorElement.className = 'alert alert-error';  // Set class
        errorElement.textContent = message;  // Set text
        
        // Add to container
        errorContainer.innerHTML = '';  // Clear container
        errorContainer.appendChild(errorElement);  // Add error element
        
        // Scroll to error
        errorContainer.scrollIntoView({ behavior: 'smooth' });  // Scroll to error
    }
});