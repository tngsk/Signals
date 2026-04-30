class NodeEditor {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) throw new Error(`Container #${containerId} not found`);

        // Setup DOM
        this.container.style.position = 'relative';
        this.container.style.overflow = 'hidden';

        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.style.position = 'absolute';
        this.svg.style.top = '0';
        this.svg.style.left = '0';
        this.svg.style.width = '100%';
        this.svg.style.height = '100%';
        this.svg.style.pointerEvents = 'none';
        this.container.appendChild(this.svg);

        // State
        this.nodes = {}; // id -> { element, data }
        this.connections = []; // { fromNode, fromPort, toNode, toPort }
        this.nodeCounter = 0;

        // Interaction state
        this.draggingNode = null;
        this.draggingPort = null;
        this.startPos = { x: 0, y: 0 };
        this.lineTemp = null;

        this._bindEvents();
    }

    _bindEvents() {
        this._onMouseDown = this._onMouseDown.bind(this);
        this._onMouseMove = this._onMouseMove.bind(this);
        this._onMouseUp = this._onMouseUp.bind(this);

        this.container.addEventListener('mousedown', this._onMouseDown);
        document.addEventListener('mousemove', this._onMouseMove);
        document.addEventListener('mouseup', this._onMouseUp);
    }

    destroy() {
        this.container.removeEventListener('mousedown', this._onMouseDown);
        document.removeEventListener('mousemove', this._onMouseMove);
        document.removeEventListener('mouseup', this._onMouseUp);
        this.container.innerHTML = '';
    }

    _onMouseDown(e) {
        const nodeEl = e.target.closest('.node');
        const portEl = e.target.closest('.port');

        if (portEl && portEl.dataset.io === 'out') {
            this.draggingPort = portEl;
            const rect = portEl.getBoundingClientRect();
            const containerRect = this.container.getBoundingClientRect();
            this.lineTemp = {
                x1: rect.left - containerRect.left + rect.width / 2,
                y1: rect.top - containerRect.top + rect.height / 2,
                x2: e.clientX - containerRect.left,
                y2: e.clientY - containerRect.top
            };
        } else if (nodeEl && !e.target.closest('input') && !e.target.closest('select') && !e.target.closest('button')) {
            this.draggingNode = nodeEl;
            const containerRect = this.container.getBoundingClientRect();
            this.startPos = {
                x: e.clientX - nodeEl.offsetLeft,
                y: e.clientY - nodeEl.offsetTop
            };
        }
    }

    _onMouseMove(e) {
        const containerRect = this.container.getBoundingClientRect();

        if (this.draggingNode) {
            this.draggingNode.style.left = (e.clientX - this.startPos.x) + 'px';
            this.draggingNode.style.top = (e.clientY - this.startPos.y) + 'px';
            this.updateLines();
        } else if (this.draggingPort) {
            this.lineTemp.x2 = e.clientX - containerRect.left;
            this.lineTemp.y2 = e.clientY - containerRect.top;
            this.updateLines();
        }
    }

    _onMouseUp(e) {
        if (this.draggingPort) {
            const targetPort = e.target.closest('.port');
            if (targetPort && targetPort.dataset.io === 'in' && targetPort.dataset.node !== this.draggingPort.dataset.node) {
                this.addConnection(
                    this.draggingPort.dataset.node,
                    this.draggingPort.dataset.port,
                    targetPort.dataset.node,
                    targetPort.dataset.port
                );
            }
            this.draggingPort = null;
            this.lineTemp = null;
            this.updateLines();
        }
        this.draggingNode = null;
    }

    addNode(type, config = {}) {
        this.nodeCounter++;
        const id = config.id || `${type}${this.nodeCounter}`;
        const x = config.x || 100;
        const y = config.y || 100;
        const name = config.name || id;

        const nodeEl = document.createElement('div');
        nodeEl.className = 'node';
        nodeEl.id = id;
        nodeEl.style.left = `${x}px`;
        nodeEl.style.top = `${y}px`;

        let contentHtml = `<div class="node-header">${name}</div>`;

        // Add default ports based on simple rules or config
        const inPorts = config.inPorts || (type === 'oscillator' ? 0 : type === 'output_wav' ? 1 : 2);
        const outPorts = config.outPorts || (type === 'output_wav' ? 0 : 1);

        for (let i = 0; i < inPorts; i++) {
            contentHtml += `<div class="port in" data-node="${id}" data-port="${i}" data-io="in" style="top: ${50 + i*20}%"></div>`;
        }
        for (let i = 0; i < outPorts; i++) {
            contentHtml += `<div class="port out" data-node="${id}" data-port="${i}" data-io="out" style="top: ${50 + i*20}%"></div>`;
        }

        // Specific inputs based on type
        if (type === 'oscillator') {
            contentHtml += `
                <select class="param" data-param="waveform">
                    <option value="sine">Sine</option>
                    <option value="square">Square</option>
                    <option value="sawtooth">Saw</option>
                </select>
                <input class="param" type="number" data-param="frequency" value="440" step="1">
            `;
        } else if (type === 'envelope_adsr') {
             contentHtml += `<div style="font-size:10px">ADSR</div>`;
        } else if (type === 'vca') {
             contentHtml += `<div style="font-size:10px">VCA</div>`;
        } else if (type === 'output_wav') {
             contentHtml += `<div style="font-size:10px">OUT</div>`;
        }

        nodeEl.innerHTML = contentHtml;
        this.container.appendChild(nodeEl);

        this.nodes[id] = {
            id,
            type,
            element: nodeEl,
            getParameters: () => {
                const params = {};
                nodeEl.querySelectorAll('.param').forEach(el => {
                    params[el.dataset.param] = el.type === 'number' ? parseFloat(el.value) : el.value;
                });
                if (type === 'output_wav') params['filename'] = 'output.wav';
                return params;
            }
        };

        return id;
    }

    addConnection(fromNode, fromPort, toNode, toPort) {
        // Prevent duplicate
        if (this.connections.some(c => c.fromNode === fromNode && c.fromPort === fromPort && c.toNode === toNode && c.toPort === toPort)) {
            return;
        }
        this.connections.push({ fromNode, fromPort, toNode, toPort });
        this.updateLines();
    }

    updateLines() {
        this.svg.innerHTML = '';
        const containerRect = this.container.getBoundingClientRect();

        this.connections.forEach(conn => {
            const fromPort = document.querySelector(`.port.out[data-node="${conn.fromNode}"][data-port="${conn.fromPort}"]`);
            const toPort = document.querySelector(`.port.in[data-node="${conn.toNode}"][data-port="${conn.toPort}"]`);

            if (fromPort && toPort) {
                const fromRect = fromPort.getBoundingClientRect();
                const toRect = toPort.getBoundingClientRect();

                this._drawLine(
                    fromRect.left - containerRect.left + fromRect.width/2,
                    fromRect.top - containerRect.top + fromRect.height/2,
                    toRect.left - containerRect.left + toRect.width/2,
                    toRect.top - containerRect.top + toRect.height/2
                );
            }
        });

        if (this.lineTemp) {
            this._drawLine(this.lineTemp.x1, this.lineTemp.y1, this.lineTemp.x2, this.lineTemp.y2);
        }
    }

    _drawLine(x1, y1, x2, y2) {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const cp1x = x1 + Math.abs(x2 - x1) * 0.5;
        const cp2x = x2 - Math.abs(x2 - x1) * 0.5;
        path.setAttribute('d', `M ${x1} ${y1} C ${cp1x} ${y1}, ${cp2x} ${y2}, ${x2} ${y2}`);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', '#888');
        path.setAttribute('stroke-width', '3');
        this.svg.appendChild(path);
    }

    getGraphData() {
        const data = { nodes: {}, connections: [] };

        for (const [id, node] of Object.entries(this.nodes)) {
            data.nodes[id] = {
                type: node.type,
                parameters: node.getParameters()
            };
        }

        this.connections.forEach(conn => {
            data.connections.push({
                from: conn.fromNode,
                fromPort: conn.fromPort,
                to: conn.toNode,
                toPort: conn.toPort
            });
        });

        return data;
    }
}
