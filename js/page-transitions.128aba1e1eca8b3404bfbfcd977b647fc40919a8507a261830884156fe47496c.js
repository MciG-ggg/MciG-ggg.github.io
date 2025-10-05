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
         * 处理链接点击的页面切换动画 - 已禁用
         */
        handleLinkTransition(event, href) {
          window.location.href = href;
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
         * 添加视差滚动效果 - 已禁用
         */
        initParallaxEffects() {
          return;
        }
        /**
         * 为特定元素添加进入视口动画 - 已禁用
         */
        initScrollAnimations() {
          const animatedElements = document.querySelectorAll("[data-animate-on-scroll]");
          animatedElements.forEach((element) => {
            element.classList.add("animate-in");
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
