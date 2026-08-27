document.addEventListener('DOMContentLoaded', () => {
  initRainEffect();
  initIsometricTilt();
  initUpload();
  initStatsCounter();
});

// 1. Falling rain particle overlay
function initRainEffect() {
  const canvas = document.getElementById('rain-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  let w = (canvas.width = window.innerWidth);
  let h = (canvas.height = window.innerHeight);
  
  const drops = [];
  const maxDrops = 80;
  
  class RainDrop {
    constructor() {
      this.reset();
    }
    
    reset() {
      this.x = Math.random() * w;
      this.y = Math.random() * -h;
      this.vy = Math.random() * 4 + 8; // Fall speed
      this.length = Math.random() * 15 + 10;
      this.weight = Math.random() * 0.5 + 0.5;
    }
    
    update() {
      this.y += this.vy;
      if (this.y > h) {
        this.reset();
      }
    }
    
    draw() {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(245, 166, 35, 0.15)'; // Very faint amber rain
      ctx.lineWidth = 1;
      ctx.moveTo(this.x, this.y);
      ctx.lineTo(this.x + 1, this.y + this.length);
      ctx.stroke();
    }
  }
  
  for (let i = 0; i < maxDrops; i++) {
    drops.push(new RainDrop());
  }
  
  function animate() {
    ctx.clearRect(0, 0, w, h);
    drops.forEach(d => {
      d.update();
      d.draw();
    });
    requestAnimationFrame(animate);
  }
  
  window.addEventListener('resize', () => {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  });
  
  animate();
}

// 2. Mouse-tracking isometric tilt
function initIsometricTilt() {
  const grid = document.getElementById('isometric-city');
  if (!grid) return;
  
  document.addEventListener('mousemove', (e) => {
    const xc = window.innerWidth / 2;
    const yc = window.innerHeight / 2;
    
    // Max 6 degrees tilt
    const tiltX = (yc - e.clientY) / yc * 6;
    const tiltY = (e.clientX - xc) / xc * 6;
    
    // RotateX is set base at 60deg, RotateZ base at -45deg
    grid.style.transform = `rotateX(${60 + tiltX}deg) rotateZ(${-45 + tiltY}deg)`;
  });
}

// 3. Upload and detect endpoints calls
function initUpload() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const loader = document.getElementById('loader');
  const scanLine = document.getElementById('scan-line');
  const preview = document.getElementById('annotated-preview');
  const idleText = document.getElementById('no-img-text');
  const mapFrame = document.getElementById('map-frame');
  const mapCounter = document.getElementById('map-counter');
  
  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleUpload(e.target.files[0]);
  });

  async function handleUpload(file) {
    loader.classList.add('active');
    scanLine.classList.add('active');
    preview.style.display = 'none';
    idleText.style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      
      if (data.success) {
        displayTelemetry(data);
      } else {
        alert("Detection failed: " + (data.detail || "Unknown error"));
        idleText.style.display = 'block';
      }
    } catch (e) {
      console.error(e);
      alert("Error contacting API server.");
      idleText.style.display = 'block';
    } finally {
      loader.classList.remove('active');
      scanLine.classList.remove('active');
    }
  }

  function displayTelemetry(data) {
    // Show annotated output
    preview.src = data.annotated_image;
    preview.style.display = 'block';
    
    // Set text parameters
    document.getElementById('val-critical').innerText = data.critical_count;
    document.getElementById('val-medium').innerText = data.medium_count;
    document.getElementById('val-small').innerText = data.small_count;
    document.getElementById('val-stretch').innerText = `${data.estimated_stretch}m`;
    
    // Set overall priority badge
    const priorityVal = document.getElementById('val-priority');
    
    let priorityText = "CLEAR";
    if (data.repair_priority === 5) {
      priorityText = "IMMEDIATE (5/5) 🔴";
    } else if (data.repair_priority === 4) {
      priorityText = "URGENT (4/5) 🔴";
    } else if (data.repair_priority === 3) {
      priorityText = "SCHEDULE (3/5) 🟡";
    } else if (data.repair_priority === 2) {
      priorityText = "ROUTINE (2/5) 🟢";
    } else if (data.repair_priority === 1) {
      priorityText = "MONITOR (1/5) 🟢";
    }
    
    priorityVal.innerText = priorityText;
    
    // Trigger map refresh to load the new markers and heatmap
    mapFrame.src = "/api/map";
    
    // Update map counter stat
    const count = parseInt(mapCounter.innerText) + 1;
    mapCounter.innerText = count;
  }
}

// 4. Counts animation
function initStatsCounter() {
  const counters = document.querySelectorAll('.count-up');
  const speed = 100;
  
  const animate = (counter) => {
    const target = +counter.getAttribute('data-target');
    const count = +counter.innerText;
    const increment = target / speed;
    
    if (count < target) {
      counter.innerText = Math.ceil(count + increment);
      setTimeout(() => animate(counter), 10);
    } else {
      counter.innerText = target.toLocaleString();
    }
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animate(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(c => observer.observe(c));
}
