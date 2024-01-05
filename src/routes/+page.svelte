<script>
    import LinePlot from './Nested.svelte';
    import * as d3 from 'd3';

    let data = d3.ticks(-2, 2, 200).map(Math.sin);

    function onMousemove(event) {
        const [x, y] = d3.pointer(event);
        data = data.slice(-200).concat(Math.atan2(x, y));
    }
</script>

<div on:mousemove={onMousemove}>
    <LinePlot {data} />
</div>
