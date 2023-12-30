<script>
    import Nested from './Nested.svelte';
    import src from '$lib/images/favicon.png';
    const multiple = [2, 3, 4];
    let count = 0;

    function increment() {
        count += 1;
    }

    $: double = count * 2;
</script>

<div class="flex min-h-screen flex-col items-center justify-center gap-4 p-4">
    <img class="w-16" {src} alt="svelte logo" />
    <h1 class="text-3xl font-bold underline"><a href="/">Welcome to SvelteKit</a></h1>
    <p class="my-4">
        Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation
    </p>

    {#if count % 2 === 0}
        <Nested />
    {:else}
        <p class="flex gap-1">
            Look at it now! <Nested kit="double" />
        </p>
    {/if}

    {#each multiple as multi}
        {@const localCount = 0}
        {@const localMulti = localCount * multi}
        {@const localIncrement = () => ()}
    
        <button
            class="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700"
            on:click={localIncrement}
        >
            Clicked {localCount}
            {localCount === 1 ? 'time' : 'times'}
            {`#click(s) x ${multi} = ${localMulti}`}
        </button>
    {/each}

    <button
        class="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700"
        on:click={increment}
    >
        Clicked {count}
        {count === 1 ? 'time' : 'times'}
        {count % 2 === 1 ? '🔥' : '🔥🔥'}
        {`#click(s) x 2 = ${double}`}
    </button>
</div>

<style lang="postcss">
    :global(html) {
        background-color: theme(colors.gray.100);
    }
</style>
