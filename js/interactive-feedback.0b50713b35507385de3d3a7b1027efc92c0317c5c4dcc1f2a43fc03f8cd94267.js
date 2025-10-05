(() => {
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __commonJS = (cb, mod) => function __require() {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  };

  // <stdin>
  var require_stdin = __commonJS({
    "<stdin>"(exports, module) {
      var InteractiveFeedback = class {
        constructor() {
          this.progressBar = null;
          this.backToTopButton = null;
          this.loadingOverlay = null;
          this.isScrolling = false;
          this.scrollTimeout = null;
          this.init();
        }
        init() {
          this.createProgressIndicator();
          this.createBackToTopButton();
          this.createLoadingOverlay();
          this.bindEvents();
        }
        /**
         * 创建阅读进度指示器
         */
        createProgressIndicator() {
          this.progressBar = document.createElement("div");
          this.progressBar.className = "reading-progress";
          this.progressBar.innerHTML = `
      <div class="reading-progress-bar"></div>
      <div class="reading-progress-circle">
        <svg class="progress-ring" width="40" height="40">
          <circle class="progress-ring-circle" 
                  cx="20" cy="20" r="16" 
                  fill="transparent" 
                  stroke="var(--accent-primary)" 
                  stroke-width="2" 
                  stroke-dasharray="100.53" 
                  stroke-dashoffset="100.53"/>
        </svg>
        <span class="progress-percentage">0%</span>
      </div>
    `;
          document.body.appendChild(this.progressBar);
        }
        /**
         * 创建返回顶部按钮
         */
        createBackToTopButton() {
          this.backToTopButton = document.createElement("button");
          this.backToTopButton.className = "back-to-top";
          this.backToTopButton.innerHTML = `
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="m18 15-6-6-6 6"/>
      </svg>
      <span class="back-to-top-text">\u8FD4\u56DE\u9876\u90E8</span>
    `;
          this.backToTopButton.addEventListener("click", () => this.scrollToTop());
          document.body.appendChild(this.backToTopButton);
        }
        /**
         * 创建加载状态遮罩
         */
        createLoadingOverlay() {
          this.loadingOverlay = document.createElement("div");
          this.loadingOverlay.className = "loading-overlay";
          this.loadingOverlay.innerHTML = `
      <div class="loading-spinner">
        <div class="spinner-ring"></div>
        <div class="spinner-ring"></div>
        <div class="spinner-ring"></div>
      </div>
      <div class="loading-text">\u52A0\u8F7D\u4E2D...</div>
    `;
          document.body.appendChild(this.loadingOverlay);
        }
        /**
         * 绑定事件监听器
         */
        bindEvents() {
          window.addEventListener("scroll", () => this.handleScroll(), { passive: true });
          window.addEventListener("load", () => this.hideLoadingOverlay());
          this.observeImageLoading();
          this.observeFormSubmissions();
          this.observeAjaxRequests();
        }
        /**
         * 处理滚动事件
         */
        handleScroll() {
          if (!this.isScrolling) {
            this.isScrolling = true;
            requestAnimationFrame(() => this.updateScrollIndicators());
          }
          if (this.scrollTimeout) {
            clearTimeout(this.scrollTimeout);
          }
          this.scrollTimeout = setTimeout(() => {
            this.isScrolling = false;
            this.hideScrollIndicators();
          }, 150);
        }
        /**
         * 更新滚动指示器
         */
        updateScrollIndicators() {
          const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
          const documentHeight = document.documentElement.scrollHeight - window.innerHeight;
          const scrollPercentage = Math.min(scrollTop / documentHeight * 100, 100);
          const progressBarElement = this.progressBar.querySelector(".reading-progress-bar");
          progressBarElement.style.width = `${scrollPercentage}%`;
          const circle = this.progressBar.querySelector(".progress-ring-circle");
          const circumference = 2 * Math.PI * 16;
          const offset = circumference - scrollPercentage / 100 * circumference;
          circle.style.strokeDashoffset = offset;
          const percentageElement = this.progressBar.querySelector(".progress-percentage");
          percentageElement.textContent = `${Math.round(scrollPercentage)}%`;
          if (scrollTop > 300) {
            this.backToTopButton.classList.add("visible");
          } else {
            this.backToTopButton.classList.remove("visible");
          }
          this.progressBar.classList.add("visible");
          this.isScrolling = false;
        }
        /**
         * 隐藏滚动指示器
         */
        hideScrollIndicators() {
          setTimeout(() => {
            if (!this.isScrolling) {
              this.progressBar.classList.remove("visible");
            }
          }, 1e3);
        }
        /**
         * 平滑滚动到顶部
         */
        scrollToTop() {
          this.backToTopButton.classList.add("clicked");
          window.scrollTo({
            top: 0,
            behavior: "smooth"
          });
          setTimeout(() => {
            this.backToTopButton.classList.remove("clicked");
          }, 300);
        }
        /**
         * 显示加载状态
         */
        showLoadingOverlay(text = "\u52A0\u8F7D\u4E2D...") {
          this.loadingOverlay.querySelector(".loading-text").textContent = text;
          this.loadingOverlay.classList.add("visible");
        }
        /**
         * 隐藏加载状态
         */
        hideLoadingOverlay() {
          this.loadingOverlay.classList.add("fade-out");
          setTimeout(() => {
            this.loadingOverlay.classList.remove("visible", "fade-out");
          }, 500);
        }
        /**
         * 观察图片加载状态
         */
        observeImageLoading() {
          const images = document.querySelectorAll('img[data-src], img[loading="lazy"]');
          if (images.length === 0) return;
          let loadedImages = 0;
          const totalImages = images.length;
          const updateLoadingProgress = () => {
            loadedImages++;
            const progress = loadedImages / totalImages * 100;
            const progressText = `\u52A0\u8F7D\u56FE\u7247\u4E2D... ${Math.round(progress)}%`;
            if (this.loadingOverlay.classList.contains("visible")) {
              this.loadingOverlay.querySelector(".loading-text").textContent = progressText;
            }
            if (loadedImages === totalImages) {
              setTimeout(() => this.hideLoadingOverlay(), 200);
            }
          };
          images.forEach((img) => {
            if (img.complete) {
              updateLoadingProgress();
            } else {
              img.addEventListener("load", updateLoadingProgress);
              img.addEventListener("error", updateLoadingProgress);
            }
          });
        }
        /**
         * 观察表单提交状态
         */
        observeFormSubmissions() {
          const forms = document.querySelectorAll("form");
          forms.forEach((form) => {
            form.addEventListener("submit", (e) => {
              this.showLoadingOverlay("\u63D0\u4EA4\u4E2D...");
              form.classList.add("submitting");
              const submitButton = form.querySelector('button[type="submit"], input[type="submit"]');
              if (submitButton) {
                submitButton.disabled = true;
                submitButton.classList.add("loading");
              }
            });
          });
        }
        /**
         * 观察AJAX请求状态
         */
        observeAjaxRequests() {
          const originalFetch = window.fetch;
          let activeRequests = 0;
          window.fetch = async (...args) => {
            activeRequests++;
            if (activeRequests === 1) {
              this.showLoadingOverlay("\u8BF7\u6C42\u4E2D...");
            }
            try {
              const response = await originalFetch(...args);
              return response;
            } finally {
              activeRequests--;
              if (activeRequests === 0) {
                setTimeout(() => this.hideLoadingOverlay(), 300);
              }
            }
          };
          const originalXHROpen = XMLHttpRequest.prototype.open;
          const originalXHRSend = XMLHttpRequest.prototype.send;
          XMLHttpRequest.prototype.open = function(...args) {
            this._url = args[1];
            return originalXHROpen.apply(this, args);
          };
          XMLHttpRequest.prototype.send = function(...args) {
            activeRequests++;
            if (activeRequests === 1) {
              this.showLoadingOverlay("\u8BF7\u6C42\u4E2D...");
            }
            this.addEventListener("loadend", () => {
              activeRequests--;
              if (activeRequests === 0) {
                setTimeout(() => this.hideLoadingOverlay(), 300);
              }
            });
            return originalXHRSend.apply(this, args);
          };
        }
        /**
         * 添加页面元素的加载动画
         */
        animateElementsOnLoad() {
          const elements = document.querySelectorAll("[data-animate-on-load]");
          elements.forEach((element, index) => {
            element.style.opacity = "0";
            element.style.transform = "translateY(20px)";
            setTimeout(() => {
              element.style.transition = "opacity 0.6s ease, transform 0.6s ease";
              element.style.opacity = "1";
              element.style.transform = "translateY(0)";
            }, index * 100 + 200);
          });
        }
        /**
         * 创建点击波纹效果
         */
        createRippleEffect(element, event) {
          const ripple = document.createElement("span");
          const rect = element.getBoundingClientRect();
          const size = Math.max(rect.width, rect.height);
          const x = event.clientX - rect.left - size / 2;
          const y = event.clientY - rect.top - size / 2;
          ripple.style.width = ripple.style.height = size + "px";
          ripple.style.left = x + "px";
          ripple.style.top = y + "px";
          ripple.classList.add("ripple-effect");
          element.appendChild(ripple);
          setTimeout(() => {
            ripple.remove();
          }, 600);
        }
        /**
         * 初始化按钮波纹效果
         */
        initButtonRipples() {
          const buttons = document.querySelectorAll('button, .btn, [role="button"]');
          buttons.forEach((button) => {
            button.addEventListener("click", (e) => {
              if (!button.classList.contains("no-ripple")) {
                this.createRippleEffect(button, e);
              }
            });
          });
        }
        /**
         * 添加键盘导航反馈
         */
        initKeyboardFeedback() {
          document.addEventListener("keydown", (e) => {
            if (e.key === "Tab") {
              document.body.classList.add("keyboard-navigation");
            }
          });
          document.addEventListener("mousedown", () => {
            document.body.classList.remove("keyboard-navigation");
          });
        }
      };
      document.addEventListener("DOMContentLoaded", () => {
        const feedback = new InteractiveFeedback();
        feedback.animateElementsOnLoad();
        feedback.initButtonRipples();
        feedback.initKeyboardFeedback();
      });
      if (typeof module !== "undefined" && module.exports) {
        module.exports = InteractiveFeedback;
      }
    }
  });
  require_stdin();
})();
