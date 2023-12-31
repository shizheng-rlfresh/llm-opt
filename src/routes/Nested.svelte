<script context="module">
    export let kk = 'sveltekit';
</script>

<script>
    /**
     * @type {string}
     */
    export let kit = 'sveltekit';
</script>

<!--
    @component
    This is a page component. hahaha   
-->

<div>
    <p class="blue">This is just a {kit} test.</p>

    <p class="blue">This is just a second {kit} test.</p>

    <p class="blue">This is just a third {kit} test.</p>

    <slot />
</div>

<style lang="postcss">
    .blue {
        background-color: var(--color, #ddd);
    }
</style>
