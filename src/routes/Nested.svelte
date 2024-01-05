<script>
    import * as d3 from 'd3';

    export let data;
    export let width = 640;
    export let height = 400;
    export let marginTop = 20;
    export let marginRight = 20;
    export let marginBottom = 30;
    export let marginLeft = 40;

    let gx;
    let gy;

    $: x = d3.scaleLinear([0, data.length - 1], [marginLeft, width - marginRight]);
    $: y = d3.scaleLinear(d3.extent(data), [height - marginBottom, marginTop]);
    $: line = d3.line((d, i) => x(i), y);
    $: d3.select(gy).call(d3.axisLeft(y));
    $: d3.select(gx).call(d3.axisBottom(x));
</script>

<svg {width} {height}>
    <g bind:this={gx} transform="translate(0,{height - marginBottom})" />
    <g bind:this={gy} transform="translate({marginLeft},0)" />
    <path fill="none" stroke="currentColor" stroke-width="1.5" d={line(data)} />
    <g fill="white" stroke="currentColor" stroke-width="1.5">
        {#each data as d, i}
            <circle key={i} cx={x(i)} cy={y(d)} r="2.5" fill="white" />
        {/each}
    </g>
</svg>
