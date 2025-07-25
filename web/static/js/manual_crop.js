class ManualCropper {
    constructor(canvasId, imageId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.image = document.getElementById(imageId);
        this.crops = [];
        this.currentCrop = null;
        this.isDrawing = false;
        this.startPoint = null;
        
        this.setupEventListeners();
        this.loadImage();
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.startCrop.bind(this));
        this.canvas.addEventListener('mousemove', this.updateCrop.bind(this));
        this.canvas.addEventListener('mouseup', this.endCrop.bind(this));
        this.canvas.addEventListener('contextmenu', this.removeLast.bind(this));
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchend', this.handleTouch.bind(this));
        
        // Keyboard events
        document.addEventListener('keydown', this.handleKeyboard.bind(this));
    }
    
    loadImage() {
        this.image.onload = () => {
            this.canvas.width = this.image.naturalWidth;
            this.canvas.height = this.image.naturalHeight;
            this.redraw();
        };
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }
    
    startCrop(e) {
        e.preventDefault();
        const pos = this.getMousePos(e);
        this.isDrawing = true;
        this.startPoint = pos;
        this.currentCrop = {
            x1: pos.x,
            y1: pos.y,
            x2: pos.x,
            y2: pos.y
        };
    }
    
    updateCrop(e) {
        if (!this.isDrawing) return;
        
        e.preventDefault();
        const pos = this.getMousePos(e);
        this.currentCrop.x2 = pos.x;
        this.currentCrop.y2 = pos.y;
        this.redraw();
    }
    
    endCrop(e) {
        if (!this.isDrawing) return;
        
        e.preventDefault();
        this.isDrawing = false;
        
        // Validate crop size
        const width = Math.abs(this.currentCrop.x2 - this.currentCrop.x1);
        const height = Math.abs(this.currentCrop.y2 - this.currentCrop.y1);
        
        if (width >= 20 && height >= 20) {
            // Normalize coordinates
            const crop = {
                x1: Math.min(this.currentCrop.x1, this.currentCrop.x2),
                y1: Math.min(this.currentCrop.y1, this.currentCrop.y2),
                x2: Math.max(this.currentCrop.x1, this.currentCrop.x2),
                y2: Math.max(this.currentCrop.y1, this.currentCrop.y2)
            };
            
            this.crops.push(crop);
            this.updateStatus(`Added crop ${this.crops.length}. Total: ${this.crops.length}`);
        } else {
            this.updateStatus('Crop too small (minimum 20x20 pixels)');
        }
        
        this.currentCrop = null;
        this.redraw();
    }
    
    removeLast(e) {
        e.preventDefault();
        if (this.crops.length > 0) {
            this.crops.pop();
            this.updateStatus(`Removed last crop. Total: ${this.crops.length}`);
            this.redraw();
        }
        return false;
    }
    
    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0] || e.changedTouches[0];
        
        if (e.type === 'touchstart') {
            this.startCrop({ clientX: touch.clientX, clientY: touch.clientY });
        } else if (e.type === 'touchmove') {
            this.updateCrop({ clientX: touch.clientX, clientY: touch.clientY });
        } else if (e.type === 'touchend') {
            this.endCrop({ clientX: touch.clientX, clientY: touch.clientY });
        }
    }
    
    handleKeyboard(e) {
        switch(e.key) {
            case 'Enter':
                this.finishCropping();
                break;
            case 'Escape':
                this.cancelCropping();
                break;
            case 'r':
            case 'R':
                this.resetCrops();
                break;
        }
    }
    
    redraw() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw image
        this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
        
        // Draw existing crops
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 2;
        this.ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
        
        this.crops.forEach((crop, index) => {
            const width = crop.x2 - crop.x1;
            const height = crop.y2 - crop.y1;
            
            this.ctx.fillRect(crop.x1, crop.y1, width, height);
            this.ctx.strokeRect(crop.x1, crop.y1, width, height);
            
            // Draw label
            this.ctx.fillStyle = '#00ff00';
            this.ctx.font = '16px Arial';
            this.ctx.fillText(`Crop ${index + 1}`, crop.x1 + 5, crop.y1 - 5);
            this.ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
        });
        
        // Draw current crop
        if (this.currentCrop && this.isDrawing) {
            this.ctx.strokeStyle = '#ff0000';
            this.ctx.lineWidth = 2;
            
            const width = this.currentCrop.x2 - this.currentCrop.x1;
            const height = this.currentCrop.y2 - this.currentCrop.y1;
            
            this.ctx.strokeRect(this.currentCrop.x1, this.currentCrop.y1, width, height);
        }
    }
    
    updateStatus(message) {
        const statusEl = document.getElementById('crop-status');
        if (statusEl) {
            statusEl.textContent = message;
        }
        console.log('Manual Crop:', message);
    }
    
    resetCrops() {
        this.crops = [];
        this.currentCrop = null;
        this.isDrawing = false;
        this.updateStatus('All crops cleared');
        this.redraw();
    }
    
    finishCropping() {
        if (this.crops.length === 0) {
            this.updateStatus('No crops selected. Create at least one crop region.');
            return;
        }
        
        this.updateStatus('Processing crops...');
        
        // Send crops to server
        const filename = new URLSearchParams(window.location.search).get('filename');
        
        fetch('/api/manual_crop', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: filename,
                crops: this.crops
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.updateStatus(`Created ${data.crops_created} crops successfully!`);
                setTimeout(() => {
                    window.location.href = data.redirect_url;
                }, 1000);
            } else {
                this.updateStatus(`Error: ${data.error}`);
            }
        })
        .catch(error => {
            this.updateStatus(`Network error: ${error.message}`);
        });
    }
    
    cancelCropping() {
        if (confirm('Cancel cropping and return to upload?')) {
            window.location.href = '/';
        }
    }
}

// Initialize cropper when page loads
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('crop-canvas')) {
        window.cropper = new ManualCropper('crop-canvas', 'source-image');
    }
});
