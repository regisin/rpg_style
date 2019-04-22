from fastai.vision import download_images, Path


"""
To retrieve a list of images from search engines:

1. Google Images:
urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));


2. DuckDuckGo (which looks like uses bing behind the scenes):
urls = Array.from(document.querySelectorAll('.tile--img__img')).map(el=>decodeURIComponent(el.src.split('=')[1]));
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
"""


path = Path('data')
for folder in ['barbarian',
                'bard',
                'cleric',
                'druid',
                'fighter',
                'monk',
                'paladin',
                'ranger',
                'rogue',
                'sorcerer',
                'warlock',
                'wizard'
                ]:
    urlfile = folder + '.csv'

    dest = path/folder
    dest.mkdir(parents=True, exist_ok=True)

    download_images(path/urlfile, dest, max_pics=200, max_workers=0)