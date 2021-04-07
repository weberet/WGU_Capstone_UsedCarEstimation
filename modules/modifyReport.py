from bs4 import BeautifulSoup, Doctype
import os

def processSaveReport():
    print('Processing and saving report')
    cwd = os.getcwd()
    folder = 'templates'
    filename = 'svreport.html'
    svreportLocation =  cwd + os.sep + folder + os.sep + filename

    try:
        soup = BeautifulSoup(open(svreportLocation),features="html.parser")
        print(f'Successfully parsed report at location: {svreportLocation}')
    except:
        print(f"Couldn't parse report at location: {svreportLocation}")

    navUpdateString = """
        //Sets nav-link to active for this page.
        $(document).ready(function () {
          $('#datagraphs').addClass('active');
        });
    """
    insertTopString = "{% extends 'base.html' %}\n{% block content %}"
    insertBottomString = "{% endblock %}"

    #Remove doctype:
    for item in soup.contents:
        if isinstance(item, Doctype):
            item.extract()
    #end remove doctype

    #Remove <html></html> tags
    for match in soup.findAll('html'):
        match.unwrap()

    #Remove <link> tags
    soup.find('link').extract()

    soup.insert(0,insertTopString)
    soup.insert(len(soup)+1, insertBottomString)

    #script insert
    scriptTag = soup.new_tag('script')
    scriptTag.append(navUpdateString)
    #soup.body.insert(len(soup.body.contents), navUpdateString)
    head = soup.find('head')
    head.insert(1,scriptTag)

    #print(soup.prettify())
    writeFile(svreportLocation,soup)

def writeFile(file,soup):
    source = open(file, "w+")
    source.write(str(soup.prettify()))
    source.close()