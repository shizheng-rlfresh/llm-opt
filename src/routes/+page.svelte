<script>
    import Nested from './Nested.svelte';
    import src from '$lib/images/favicon.png';
    let count = 0;

    function increment() {
        count += 1;
    }

    $: double = count * 2;
</script>

<img class="w-16" {src} alt="svelte logo" />
<h1 class="text-3xl font-bold underline"><a href="/">Welcome to SvelteKit</a></h1>
<p class="my-4">
    Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation
</p>

<Nested />

<button
    class="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700"
    on:click={increment}
>
    Clicked {count}
    {count === 1 ? 'time' : 'times'}
    {count % 2 === 1 ? '🔥' : '🔥🔥'}
    {`#click(s) x 2 = ${double}`}
</button>

<style lang="postcss">
    :global(html) {
        background-color: theme(colors.gray.100);
    }
</style>
