import type { OverlayProxy } from 'embedding-atlas/svelte';

const MARKER_SIZE = 16;
const MARKER_COLOR = '#ffffff';
const MARKER_SHADOW = '#000000';

/**
 * Creates a CustomOverlay class that renders a crosshair marker at a fixed data-space
 * position. The marker repositions on every pan/zoom via proxy.location().
 */
export function createSearchMarkerOverlay(dataX: number, dataY: number) {
    return class {
        private el: HTMLDivElement;

        constructor(node: HTMLDivElement, { proxy }: { proxy: OverlayProxy }) {
            this.el = document.createElement('div');
            this.el.style.cssText = `
                position: absolute;
                width: ${MARKER_SIZE}px;
                height: ${MARKER_SIZE}px;
                pointer-events: none;
                transform-origin: center center;
            `;
            this.el.innerHTML = buildCrosshairSVG();
            node.appendChild(this.el);
            this.reposition(proxy);
        }

        update({ proxy }: { proxy: OverlayProxy }) {
            this.reposition(proxy);
        }

        destroy() {
            this.el.remove();
        }

        private reposition(proxy: OverlayProxy) {
            const { x, y } = proxy.location(dataX, dataY);
            this.el.style.left = `${x - MARKER_SIZE / 2}px`;
            this.el.style.top = `${y - MARKER_SIZE / 2}px`;
        }
    };
}

function buildCrosshairSVG(): string {
    const s = MARKER_SIZE;
    const half = s / 2;
    const strokeWidth = 2;
    const shadowWidth = strokeWidth + 2;
    return `
        <svg width="${s}" height="${s}" viewBox="0 0 ${s} ${s}" xmlns="http://www.w3.org/2000/svg">
            <!-- shadow layer for contrast -->
            <line x1="${half}" y1="0" x2="${half}" y2="${s}"
                stroke="${MARKER_SHADOW}" stroke-width="${shadowWidth}" stroke-linecap="round" opacity="0.6"/>
            <line x1="0" y1="${half}" x2="${s}" y2="${half}"
                stroke="${MARKER_SHADOW}" stroke-width="${shadowWidth}" stroke-linecap="round" opacity="0.6"/>
            <!-- main crosshair -->
            <line x1="${half}" y1="0" x2="${half}" y2="${s}"
                stroke="${MARKER_COLOR}" stroke-width="${strokeWidth}" stroke-linecap="round"/>
            <line x1="0" y1="${half}" x2="${s}" y2="${half}"
                stroke="${MARKER_COLOR}" stroke-width="${strokeWidth}" stroke-linecap="round"/>
            <!-- center dot -->
            <circle cx="${half}" cy="${half}" r="2.5"
                fill="${MARKER_COLOR}" stroke="${MARKER_SHADOW}" stroke-width="1" opacity="0.9"/>
        </svg>
    `;
}
