(() => {
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __commonJS = (cb, mod) => function __require() {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  };

  // <stdin>
  var require_stdin = __commonJS({
    "<stdin>"(exports, module) {
      var PageTransitions = class {
        constructor() {
          this.isLoading = false;
          this.themeTransitionOverlay = null;
          this.routeLoadingIndicator = null;
          this.init();
        }
        init() {
          this.createTransitionElements();
          this.bindEvents();
          this.initPageLoad();
        }
        /**
         * 创建过渡动画所需的DOM元素
         */
        createTransitionElements() {
          this.themeTransitionOverlay = document.createElement("div");
          this.themeTransitionOverlay.className = "theme-transition-overlay";
          document.body.appendChild(this.themeTransitionOverlay);
          this.routeLoadingIndicator = document.createElement("div");
          this.routeLoadingIndicator.className = "route-loading-indicator";
          document.body.appendChild(this.routeLoadingIndicator);
        }
        /**
         * 绑定事件监听器
         */
        bindEvents() {
          if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", () => this.handlePageLoad());
          } else {
            this.handlePageLoad();
          }
          this.observeThemeChanges();
          this.bindLinkTransitions();
          window.addEventListener("popstate", () => this.handleRouteChange());
        }
        /**
         * 初始化页面加载动画 - 已禁用
         */
        initPageLoad() {
          return;
        }
        /**
         * 处理页面加载完成 - 已禁用动画
         */
        handlePageLoad() {
          return;
        }
        /**
         * 渐进显示内容元素 - 已禁用
         */
        animateContentElements() {
          return;
        }
        /**
         * 观察主题变化
         */
        observeThemeChanges() {
          const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
              if (mutation.type === "attributes" && mutation.attributeName === "data-scheme") {
                this.handleThemeTransition();
              }
            });
          });
          observer.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ["data-scheme"]
          });
          const themeToggle = document.querySelector("[data-toggle-theme]");
          if (themeToggle) {
            themeToggle.addEventListener("click", () => {
              this.handleThemeTransition();
            });
          }
        }
        /**
         * 处理主题切换动画 - 已禁用
         */
        handleThemeTransition() {
          return;
        }
        /**
         * 绑定链接过渡动画
         */
        bindLinkTransitions() {
          const internalLinks = document.querySelectorAll('a[href^="/"], a[href^="./"], a[href^="../"]');
          internalLinks.forEach((link) => {
            link.addEventListener("click", (e) => {
              if (!link.href.includes("#") && !link.target) {
                this.handleLinkTransition(e, link.href);
              }
            });
          });
        }
        /**
         * 处理链接点击的页面切换动画
         */
        handleLinkTransition(event, href) {
          if (this.isLoading) return;
          this.showRouteLoadingIndicator();
          const mainContent = document.querySelector("main");
          if (mainContent) {
            mainContent.classList.add("page-exit-active");
          }
          setTimeout(() => {
            window.location.href = href;
          }, 300);
        }
        /**
         * 显示路由加载指示器
         */
        showRouteLoadingIndicator() {
          this.isLoading = true;
          this.routeLoadingIndicator.classList.add("loading");
          setTimeout(() => {
            this.hideRouteLoadingIndicator();
          }, 3e3);
        }
        /**
         * 隐藏路由加载指示器
         */
        hideRouteLoadingIndicator() {
          this.isLoading = false;
          this.routeLoadingIndicator.classList.remove("loading");
        }
        /**
         * 处理路由变化
         */
        handleRouteChange() {
          this.showRouteLoadingIndicator();
          setTimeout(() => {
            this.handlePageLoad();
            this.hideRouteLoadingIndicator();
          }, 100);
        }
        /**
         * 添加视差滚动效果
         */
        initParallaxEffects() {
          const parallaxElements = document.querySelectorAll(".parallax-element");
          if (parallaxElements.length === 0) return;
          let ticking = false;
          const updateParallax = () => {
            const scrollY = window.pageYOffset;
            parallaxElements.forEach((element) => {
              const speed = element.dataset.parallaxSpeed || 0.5;
              const yPos = -(scrollY * speed);
              element.style.transform = `translateY(${yPos}px)`;
            });
            ticking = false;
          };
          const requestTick = () => {
            if (!ticking) {
              requestAnimationFrame(updateParallax);
              ticking = true;
            }
          };
          window.addEventListener("scroll", requestTick, { passive: true });
        }
        /**
         * 为特定元素添加进入视口动画
         */
        initScrollAnimations() {
          const animatedElements = document.querySelectorAll("[data-animate-on-scroll]");
          if (animatedElements.length === 0) return;
          const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
              if (entry.isIntersecting) {
                entry.target.classList.add("animate-in");
                observer.unobserve(entry.target);
              }
            });
          }, {
            threshold: 0.1,
            rootMargin: "0px 0px -50px 0px"
          });
          animatedElements.forEach((element) => {
            observer.observe(element);
          });
        }
      };
      document.addEventListener("DOMContentLoaded", () => {
        new PageTransitions();
      });
      if (typeof module !== "undefined" && module.exports) {
        module.exports = PageTransitions;
      }
    }
  });
  require_stdin();
})();
