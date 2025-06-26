---
layout: post
title: Experiments in Constrained LLM Decoding
description: Writing with only the 1,000 most common words in the English language.
authors: [tlyleung]
x: 93
y: 94
---

Randall Munroe's *Thing Explainer*[^thing-explainer] is a book about "complicated stuff in simple words". In it, he explains everything from nuclear reactors to data centers using only the 1,000 most common English words (or ten hundred words as he puts it). It's full of phrases like "machine that fly" instead of "airplane" or "heavy hot rock" instead of "nuclear fuel". By now, we know that LLMs are powerful, so can we use them to help us write with a limited vocabulary?

First, we need the list of allowed words. Munroe published the [list of common words](https://xkcd.com/simplewriter/words.js) used in *Thing Explainer*, and you can even try your hand at writing with it using his [Simple Writer editor](https://xkcd.com/simplewriter/), that marks banned words in red.

Let's go ahead and grab that list and put the words in a Python list.



```python
VOCAB = "understandings|understanding|conversations|disappearing|informations|grandmothers|grandfathers|questionings|conversation|information|approaching|understands|immediately|positioning|questioning|grandmother|travellings|questioners|recognizing|recognizers|televisions|remembering|rememberers|expressions|discovering|disappeared|interesting|grandfather|straightest|controllers|controlling|considering|remembered|cigarettes|companying|completely|spreadings|considered|continuing|controlled|stationing|controller|straighter|stretching|businesses|somebodies|soldiering|countering|darknesses|situations|directions|disappears|younglings|suggesting|afternoons|breathings|distancing|screenings|schoolings|especially|everything|everywhere|explaining|explainers|expression|branchings|revealings|repeatings|surprising|rememberer|somewheres|television|themselves|recognizer|recognizes|recognized|belongings|finishings|travelling|questioner|beginnings|travelings|questioned|followings|pretending|forgetting|forgetters|forwarding|positioned|travellers|gatherings|perfecting|understand|understood|weightings|approaches|officering|numberings|happenings|mentioning|letterings|husbanding|imaginings|approached|apartments|whispering|interested|discovered|spinnings|clearings|climbings|spendings|clothings|colorings|soundings|truckings|somewhere|troubling|companies|companied|beautiful|computers|confusing|considers|travelers|youngling|continues|continued|traveller|traveling|yellowing|apartment|beginning|wheelings|travelled|sometimes|something|appearing|cornering|believing|countered|believers|countries|soldiered|coverings|creatures|crossings|accepting|daughters|belonging|situation|silvering|different|silencing|touchings|bettering|tomorrows|disappear|thinkings|boardings|discovers|admitting|wrappings|distances|distanced|sightings|shrugging|doctoring|showering|shoulders|shoppings|shootings|dressings|sheetings|shadowing|settlings|servicing|seriously|seconding|searching|weighting|screening|screaming|schooling|teachings|bothering|everybody|botherers|bottoming|excepting|expecting|explained|direction|explainer|surprised|surprises|waterings|branching|revealing|returning|surfacing|familiars|repeating|fathering|reminding|supposing|breasting|attacking|remembers|breathing|remaining|breathers|brightest|brownings|suggested|recognize|fightings|attention|figurings|receiving|reasoning|realizing|fingering|buildings|finishing|stupidest|stuffings|questions|watchings|flashings|strongest|strikings|flighting|flowering|promisers|promising|following|bathrooms|prettiest|pretended|stretched|foreheads|foresting|stretches|forgotten|pressings|forgetter|strangest|preparing|forwarded|strangers|possibles|positions|afternoon|straights|pocketing|gardening|pleasings|wondering|gathering|picturing|personals|perfected|stomaches|stomached|carefully|stationed|catchings|parenting|paintings|orderings|groupings|wintering|officered|offerings|centering|numbering|neighbors|certainly|happening|narrowing|narrowest|mountains|mothering|mirroring|middlings|messaging|standings|mentioned|mattering|marriages|histories|machining|hospitals|listening|lightings|springing|lettering|husbanded|spreaders|whispered|imagining|imaginers|spreading|important|languages|answering|cigarette|interests|spiriting|cleanings|knockings|soundest|coatings|sounders|sounding|colleges|coloring|colorful|wouldn't|training|colorers|sorriest|worrying|belonged|approach|tracking|touchers|touching|computer|whatever|toppings|confused|confuses|workings|consider|bettered|teething|tonights|tonguers|tonguing|continue|arriving|tomorrow|controls|together|blacking|blackest|throwers|blocking|throwing|coolings|someones|blockers|somebody|thirties|soldiers|cornered|weighted|counting|thoughts|counters|thinking|thinners|thinning|coursing|covering|thinnest|craziest|snapping|creating|creature|thickest|boarding|crossing|smokings|crowding|smelling|smallest|cuttings|slipping|slightly|dancings|sleepers|sleeping|slamming|wordings|darkness|daughter|boatings|skinning|weddings|thanking|sittings|deciding|deciders|singling|singings|despites|simplest|terrible|silvered|tellings|wearings|youngest|watering|silences|teachers|bookings|agreeing|teaching|discover|attacked|bothered|botherer|watching|swingers|bottling|distance|silenced|signings|bottomed|sighting|shutting|shrugged|wondered|swinging|doctored|sweetest|showered|showings|doorways|shouting|shoulder|wronging|shortest|surprise|dragging|shopping|shooters|drawings|actually|shooting|dreaming|dressing|avoiding|shitting|shirting|shipping|drinking|drinkers|braining|sheeting|sharpest|drivings|sharpers|dropping|droppers|shadowed|surfaced|settling|washings|settings|services|serviced|earliest|backings|earthing|servings|branches|branched|seconded|seatings|surfaces|searched|searches|walkings|screened|waitings|screamed|supposed|emptiest|emptying|breaking|breakers|schooled|enjoying|enjoyers|entering|runnings|breasted|rounders|rounding|supposes|everyone|visitors|visiting|breathed|excepted|roofings|exciting|breathes|expected|rollings|bankings|breather|explains|villages|bridging|viewings|brighter|ringings|righting|suitings|bringing|revealed|bringers|returned|failings|repliers|replying|repeated|brothers|familiar|wintered|families|suggests|farthest|furthest|browning|fathered|removing|building|reminded|bathroom|allowing|suddenly|remember|allowers|feedings|builders|burnings|feelings|remained|refusing|stupider|windings|although|stuffing|studying|business|angriest|fighting|fighters|students|figuring|received|twenties|receives|fillings|reasoned|findings|stronger|turnings|realizes|realized|readiest|fingered|readying|striking|trusters|finishes|trusting|finished|readings|reachers|reaching|quieters|quietest|quieting|fittings|quickest|writings|beaching|question|trucking|callings|stranger|flashing|beatings|answered|flattest|flatting|flighted|straight|troubled|flowered|pullings|storming|promiser|couldn't|promised|promises|couldn't|followed|stoppers|problems|probably|prettier|stopping|pretends|stomachs|troubles|pressers|tripping|forehead|stickers|forested|pressing|whispers|carrying|sticking|carriers|stepping|stealers|forwards|stealing|becoming|prepares|prepared|powering|freeings|stations|possible|position|freshest|beddings|wrapping|fronting|catching|fuckings|policing|funniest|pointers|pointing|catchers|pocketed|gardened|starters|ceilings|pleasing|gathered|starting|centered|platings|plastics|planning|pictured|pictures|traveler|pickings|personal|glancing|yourself|chancing|perfects|changing|peopling|partying|partings|parented|grabbing|grabbers|changers|checking|starring|bedrooms|checkers|pairings|standing|painting|outsides|greatest|cheeking|greening|greenest|grouping|ordering|anything|openings|guarding|wheeling|officers|guessing|spreader|offering|children|anywhere|numbered|choicest|noticers|noticing|hallways|nothings|hangings|nobodies|admitted|neighbor|choosing|choosers|happened|neckings|happiest|narrowed|narrower|spotting|churches|mouthing|traveled|mountain|mothered|accepted|mornings|mirrored|headings|spirited|hearings|heatings|circling|middling|messaged|messages|heaviest|wouldn't|spinners|mentions|helpings|cleanest|memories|meetings|meanings|appeared|mattered|marrieds|marrying|marriage|yellowed|markings|cleaning|managing|cleaners|holdings|machined|machines|lunching|luckiest|lowering|longings|clearest|hospital|lockings|littlest|clearing|listened|housings|lightest|lighting|lighters|spinning|hundreds|hurrying|believes|spenders|believed|climbing|husbands|lettered|lettings|learning|leadings|ignoring|laughing|ignorers|imagines|yellower|imagined|climbers|imaginer|spending|closings|specials|speakers|language|believer|clothing|clouding|speaking|interest|spacings|landings|knowings|southest|jacketed|knocking|kitchens|kissings|killings|keepings|dresses|biggest|sticker|careful|shirted|warmers|shipped|birding|drinker|carries|sheeted|warming|carried|carrier|driving|sharper|tonight|drivers|casings|sharers|sharing|stepped|dropped|dropper|whisper|shapers|shaping|shakers|shaking|tonguer|shadows|stealer|several|tongued|staying|settles|settled|dusting|setting|tongues|catting|backing|catches|earlier|warmest|earthed|service|serving|warring|wanters|catcher|serious|eastest|sensing|senders|easiest|sending|sellers|selling|seeming|seeings|tiniest|seconds|station|causing|seating|edgings|stating|timings|efforts|starter|causers|screens|blacker|ceiling|screams|centers|wanting|walling|walkers|certain|emptied|empties|emptier|thrower|endings|started|schools|scarers|scaring|sayings|engines|savings|sanding|enjoyed|starers|saddest|enjoyer|staring|enoughs|rushing|bagging|runners|entered|running|chances|entires|chancer|rubbing|rowings|rounder|chanced|rounded|starred|rooming|changed|changes|blocked|angrier|exactly|changer|blocker|excepts|checked|excited|walking|excites|roofing|through|expects|blooded|checker|cheeked|throats|explain|wakings|springs|thought|waiting|blowing|rolling|rocking|risings|ringing|baggers|animals|righter|righted|ridings|richest|facings|reveals|blowers|choicer|choices|returns|voicing|worries|resting|chooses|failing|spreads|replier|failers|falling|spotted|replies|replied|chooser|thinned|fallers|thinner|balling|boarded|repeats|visitor|farther|further|circles|another|removed|fastest|removes|fathers|thicker|circled|visited|reminds|fearing|spirits|classes|answers|banking|boating|cleaned|feeding|spinner|thanked|village|worried|feeling|cleaner|remains|cleared|refuses|refused|workers|reddest|telling|yellows|spender|working|clearer|clearly|climbed|tearing|fighter|teaming|figured|figures|booking|viewing|climber|usually|closest|receive|filling|teacher|reasons|closing|finally|closers|anybody|finding|anymore|realize|special|finders|booting|realest|clothed|readier|readies|readied|fingers|teaches|tallest|clothes|speaker|readers|talkers|clouded|talking|reading|firings|spacing|takings|reacher|reached|coating|reaches|raising|raining|fishing|quietly|fittest|fitting|systems|whether|bothers|wrapped|fitters|quieted|quieter|quickly|coffees|quicker|fixings|coldest|sounded|sounder|actings|anyways|college|flashed|flashes|bottles|flatter|flatted|colored|bottled|wording|turning|sorting|flights|colorer|putting|pushers|pushing|flowers|pullers|swinger|wonders|sorrier|pulling|proving|comings|bottoms|promise|truster|boxings|company|follows|younger|trusted|sweeter|yelling|problem|without|beached|footing|confuse|beaches|brained|bearing|pretend|trucked|forcing|presser|wishing|trouble|forests|appears|beating|airings|forever|surface|control|forgets|accepts|pressed|wronged|winters|forming|presses|prepare|beaters|breaker|wheeled|because|forward|coolers|cooling|allowed|powered|pourers|freeing|pouring|tripped|coolest|breasts|someone|fresher|suppose|somehow|friends|breaths|copping|fronted|becomes|porches|poppers|popping|poorest|treeing|fucking|fullest|pooling|breathe|polices|funnier|funnies|policed|bedding|corners|futures|pointer|pointed|gamings|counted|soldier|pockets|wetting|pleased|gardens|wetters|wettest|pleases|counter|sunning|players|westest|country|gathers|bridges|playing|plating|bridged|plastic|couples|softest|getting|planned|getters|placing|gifting|pinking|pilings|piecing|picture|coursed|courses|summers|picking|snowing|phoning|bedroom|glances|glanced|winging|snapped|glassed|glasses|perhaps|covered|crazies|crazier|perfect|peopled|persons|peoples|suiting|pausing|passing|goldest|partied|windows|parties|parting|creates|grabbed|smokers|created|grabber|brought|weights|bringer|arrives|crosser|crosses|grasses|parents|palming|graying|pairing|crossed|painted|arrived|greying|smoking|paining|outside|brother|greater|smilers|outings|greened|greener|crowded|travels|smiling|ordered|grounds|offings|smelled|openers|browner|grouped|opening|smaller|growing|okaying|officer|guarded|slowest|slowing|cupping|slipped|guessed|guesses|cutting|offices|gunning|offered|browned|allower|nursing|numbing|suggest|cutters|numbers|sliders|halving|sliding|noticer|wedding|notices|noticed|nothing|writers|hallway|handing|sleeper|normals|noising|hanging|nodding|dancing|wearing|writing|slammed|hangers|darkest|skinned|happens|trained|needing|builder|beliefs|happier|necking|nearest|hardest|nearing|burning|believe|winding|hatting|narrows|stupids|sitting|mouthed|deadest|watered|sisters|mothers|singled|winning|morning|mooning|moments|heading|missing|decides|decided|decider|mirrors|minutes|hearing|minings|already|minding|middled|heating|burners|singles|middles|deepest|stuffed|heaters|singing|simpler|heavier|heavies|belongs|message|despite|mention|simples|studies|studied|silvers|helping|helpers|members|meeting|willing|meanest|attacks|herself|meaning|dinners|student|hidings|matters|marries|married|busying|busiest|silence|against|highest|wildest|hilling|marking|mapping|manages|managed|himself|history|tracked|strikes|manning|hitting|makings|hitters|whiting|towards|watched|holding|toucher|machine|holders|lunches|lunched|watches|luckier|stretch|streets|lowered|loudest|lookers|looking|longing|calling|longest|locking|bending|washing|signing|hottest|littler|benders|strange|sighted|listens|linings|likings|housing|beneath|sighing|sicking|however|lighted|sickest|lighter|calming|lifters|hundred|calmest|hurried|hurries|lifting|touched|doesn't|doesn't|hurting|touches|showers|husband|doctors|letters|cameras|letting|tossing|leaving|learned|dogging|leaning|leafing|leaders|leading|whitest|layered|ignored|showing|ignores|stories|ignorer|shoving|laughed|lasting|largest|imaging|doorway|besting|imagine|shouted|stormed|downing|storing|topping|avoided|dragged|shorter|betters|stopper|landers|insides|instead|written|drawing|shopped|stopped|between|landing|shooter|knowing|jackets|dreamed|carding|toothed|knocked|knifing|kitchen|joining|teethed|stomach|joiners|kissing|kindest|killers|killing|shoeing|kidding|jumping|kickers|kicking|jumpers|keepers|dressed|keeping|enough|checks|kicked|jumper|kicker|kidded|jumped|killed|joking|killer|kinder|joiner|kisses|kissed|joined|knives|knifes|knifed|jacket|knocks|itself|ladies|landed|lander|inside|larger|images|lasted|imaged|laughs|ignore|aboves|laying|accept|layers|across|yellow|leaded|leader|leaved|leaned|learns|leaves|yelled|lesser|letter|living|lifted|lifter|humans|hugest|lights|wrongs|houses|liking|likers|lining|housed|acting|listen|hotels|little|hotter|locals|locked|horses|longer|longed|looked|hoping|looker|losing|adding|louder|loving|lovers|lowing|lowest|writer|lowers|homing|holing|holder|making|hitter|makers|manned|manage|writes|admits|mapped|marked|hilled|higher|afraid|hiding|hidden|matter|ageing|helper|member|helped|memory|hellos|heater|metals|middle|heated|mights|minded|hearts|mining|minute|headed|mirror|misses|missed|moment|moneys|monies|months|mooned|mostly|having|mother|worlds|hating|mouths|moving|movers|movies|musics|worker|myself|naming|namers|narrow|hatted|hardly|nearer|neared|nearly|harder|necked|needed|happen|hanger|newest|nicest|nights|worked|nobody|nodded|handed|noises|noised|worded|normal|norths|nosing|agrees|noting|notice|halves|halved|number|guying|numbed|nurses|nursed|agreed|wooden|offing|gunned|offers|office|guards|wonder|okayed|okay'd|okay'd|ok'ing|ok'ing|oldest|womens|opened|opener|groups|womans|within|ground|orders|others|outing|wished|greens|greats|owning|wishes|owners|paging|pained|paints|greyed|greyer|paired|palest|grayed|palmed|papers|grayer|parent|parted|passed|golder|passes|pauses|paused|paying|person|people|wipers|goings|glance|phones|phoned|photos|picked|giving|givens|pieces|pieced|piling|gifted|pinked|pinker|places|placed|getter|gotten|plated|plates|gently|played|gather|player|please|gating|garden|pocket|gamers|points|pointy|gaming|future|wiping|fuller|police|pooled|poorer|fucked|popped|popper|fronts|friend|freers|poured|pourer|freest|powers|formed|forget|forgot|forest|forces|forced|footed|pretty|follow|fliers|flyers|proven|airing|proves|proved|prover|pulled|flying|puller|flower|pushes|pushed|floors|pusher|flight|fixers|fixing|quicks|winter|fitted|quiets|fitter|winged|radios|rained|raises|raised|fishes|rather|fished|firsts|firing|reader|finish|finger|fining|finest|realer|finder|really|finals|reason|filled|figure|fought|fights|fields|fewest|redder|refuse|remain|feeing|remind|feared|father|faster|remove|repeat|family|faller|fallen|failer|failed|rested|fading|return|reveal|riches|richer|riding|ridden|window|riders|rights|facing|allows|ringed|rising|rivers|extras|rocked|rolled|expect|roofed|excite|except|rooves|roomed|events|rounds|rowing|evened|rubbed|almost|entire|runner|enters|keying|rushed|rushes|sadder|safest|sanded|enjoys|saving|engine|savers|winded|saying|enders|scared|scares|scarer|scenes|ending|school|scream|either|eights|screen|egging|effort|search|edging|seated|second|eaters|seeing|seemed|eating|seller|sender|senses|sensed|easier|easily|earths|serves|served|willed|dusted|settle|during|driers|sevens|sexing|shadow|shakes|shaken|dryers|shaker|always|shaped|driest|shapes|shaper|drying|shares|shared|sharer|sharps|driver|drives|driven|sheets|droves|drinks|shirts|drunks|shoots|dreams|shorts|dozens|should|downed|shouts|shoved|shoves|showed|wilder|shower|dogged|doctor|shrugs|didn't|sicker|sicked|didn't|siding|sighed|doings|sights|signed|dinner|silent|silver|dyings|widest|simple|simply|deeper|single|decide|deaths|sister|deader|sizing|darker|wholes|sleeps|dances|danced|slides|slider|cutter|slower|slowed|slowly|smalls|cupped|smells|smelly|crying|smiles|smiled|smiler|crowds|smokes|smoked|smoker|create|covers|snowed|whited|softer|course|softly|couple|counts|corner|whiter|copped|cooled|cooler|coming|whites|sorted|colors|colder|sounds|coffee|coated|spaces|clouds|spaced|spoken|speaks|clothe|closed|closes|closer|spends|climbs|clears|cleans|spirit|cities|circle|church|choose|spread|chosen|choice|chests|sprung|spring|sprang|stages|stairs|cheeks|stands|keeper|change|chance|stared|stares|starer|chairs|starts|center|causer|caused|states|stated|causes|caught|catted|stayed|steals|stolen|casing|sticks|caring|carded|stones|animal|cannot|stored|stores|storms|answer|camera|calmer|calmed|called|street|buyers|bought|strike|struck|buying|anyone|strong|busier|busied|busing|burner|stuffs|burned|stupid|builds|browns|suites|suited|brings|summer|bright|sunned|bridge|breath|breast|breaks|broken|surest|branch|brains|anyway|boxing|wheels|sweets|swings|bottom|bottle|system|bother|tables|taking|takers|talked|talker|boring|taller|booted|taught|booked|teamed|teared|boning|appear|bodies|thanks|boated|thicks|boards|bluest|things|thinks|blower|thirds|thirty|though|threes|throat|bloods|thrown|throws|blocks|timing|blacks|timers|tinier|biters|tiring|todays|biting|toning|tongue|arming|birded|bigger|wetter|toothy|beyond|better|topped|tossed|bested|tosses|beside|bender|toward|bended|tracks|belong|trains|belief|travel|behind|begins|before|bedded|became|become|beater|beaten|trucks|truest|aren't|aren't|trusts|truths|trying|turned|twenty|around|uncles|weight|wasn't|wasn't|arrive|unless|upping|wedded|viewed|barely|visits|banked|balled|voices|voiced|waited|bagger|waking|walked|bagged|walker|walled|asking|wanted|wanter|warred|waring|backed|warmed|warmer|babies|washed|washes|avoids|attack|waters|asleep|watery|waving|wavers|seems|party|minds|eaten|sells|sends|known|sense|hours|pasts|paths|easts|pause|mined|layer|payed|serve|earth|early|wills|aired|heard|hears|dusts|kills|goers|hotel|seven|dried|ideas|sexed|sexes|going|drier|dries|dryer|glass|heads|shake|leads|shook|aging|gives|phone|local|photo|shape|picks|above|locks|money|drops|share|given|wrong|girls|month|sharp|piece|wilds|sheet|drove|drive|moons|lands|piles|ships|drink|piled|drank|drunk|shirt|pinks|shits|dress|shoes|mores|shoot|longs|shots|dream|drawn|draws|drags|shops|haves|horse|short|gifts|dozen|place|downs|shout|hopes|shove|hoped|plans|wiper|doors|shown|shows|wiped|plate|world|mouth|doers|joins|shrug|shuts|leafs|moved|plays|moves|sicks|don't|pleas|sided|sides|sighs|don't|gated|sight|looks|gates|wives|mover|signs|doing|dirts|knees|movie|learn|gamer|games|gamed|dying|music|since|desks|sings|singe|deeps|point|acted|musts|yells|funny|death|wider|loses|sixes|whose|names|sizes|sized|skins|keyed|skies|pools|slams|darks|named|slept|namer|sleep|leave|dance|slide|hated|young|whole|fucks|who's|slips|who's|slows|front|porch|loved|hates|small|fresh|cries|cried|smell|white|nears|loves|smile|freer|pours|lover|freed|power|smoke|frees|yeses|crowd|cross|jokes|fours|snaps|crazy|forms|cover|homed|snows|among|necks|happy|least|press|force|homes|count|needs|wipes|years|cools|foots|joked|foods|never|songs|comes|sorry|flier|color|sorts|souls|lower|newer|flyer|colds|sound|flown|south|works|coats|space|nicer|prove|lucky|spoke|night|speak|cloud|hurts|yards|pulls|holed|flies|close|climb|spent|spend|words|holes|hangs|clear|lunch|spins|clean|class|liars|floor|holds|spots|alive|noise|flats|chose|flash|nones|child|fixer|fixed|fixes|chest|cheek|mains|stage|hands|makes|stair|quick|stood|check|fiver|stand|stars|fives|north|wrote|stare|lying|quiet|noses|quite|start|chair|nosed|radio|lived|rains|notes|state|large|cause|raise|catch|noted|maker|stays|halls|angry|stole|steal|reach|first|cased|cases|steps|lives|fires|stuck|carry|stick|cares|still|cared|fired|cards|added|stone|reads|halve|stops|write|can't|ready|hairy|store|hairs|can't|storm|numbs|story|could|finer|knife|fines|calms|fined|calls|hurry|while|buyer|finds|nurse|found|which|lifts|admit|final|fills|lasts|keeps|where|buses|bused|study|offed|stuff|fight|woods|burnt|burns|field|human|build|built|wings|offer|brown|allow|guyed|suite|suits|bring|marks|fewer|feels|hills|wines|later|feeds|agree|guess|surer|fears|broke|break|guard|brain|highs|often|marry|ahead|knock|boxes|sweet|boxed|okays|swing|swung|falls|reply|hides|fails|huger|table|takes|taken|laugh|taker|rests|house|talks|bored|women|faded|fades|wheel|facts|wraps|boots|teach|faces|teams|older|books|tears|bones|maybe|woman|faced|areas|boned|opens|tells|rides|grows|thank|their|boats|thens|there|these|thick|rider|after|board|right|bluer|thins|blues|blued|grown|thing|again|rings|think|blows|blown|third|would|means|those|risen|three|rises|blood|eying|heres|throw|block|threw|roses|group|river|black|tying|times|timed|roads|rocks|order|timer|meant|green|tired|tires|extra|meets|today|rolls|biter|bitey|other|toned|tones|light|bites|worry|birds|roofs|armed|outer|rooms|outed|every|tooth|teeth|round|image|bests|event|liked|evens|rowed|likes|touch|bends|windy|bents|towns|winds|great|below|track|overs|owned|liker|train|enter|wound|begun|helps|began|begin|owner|beers|kinds|wests|paged|trees|treed|tripe|trips|pages|alone|hello|beats|enjoy|bears|truck|beach|safer|trues|truer|trued|safes|hells|sames|trust|truth|pains|wells|sands|tried|tries|greys|turns|isn't|isn't|heavy|twice|saves|uncle|saved|under|kicks|saver|paint|lines|grays|until|weeks|upped|pairs|using|asked|usual|scare|being|ender|metal|views|paled|banks|visit|pales|paler|voice|scene|heats|waits|balls|ended|empty|woken|palms|wakes|waked|walks|lined|knows|pants|worse|paper|walls|worst|wants|eight|heart|along|backs|egged|jumps|warms|grass|might|edges|grabs|seats|avoid|parts|edged|aunts|watch|about|eater|won't|water|won't|waved|waves|goods|waver|golds|wears|ears|grab|fits|each|sets|knee|lots|part|dust|noes|fish|stay|good|rain|cats|work|wild|laid|hang|gold|pass|step|loud|case|help|your|past|nods|home|care|path|hell|read|love|fire|gods|lift|card|stop|pays|keys|cars|paid|idea|fine|none|real|into|drop|heat|wish|cans|kids|find|goer|goes|went|calm|just|lead|gone|call|fill|nose|ship|huge|acts|lows|buys|some|note|kind|shit|shat|mind|ices|busy|pick|hand|shod|shoe|gave|reds|shot|hall|fews|ours|feel|burn|drew|such|draw|shop|give|felt|wing|suit|drag|hear|feed|mine|girl|feds|iced|down|when|fees|half|suns|able|word|fear|nows|door|fast|sure|leaf|pile|jobs|show|wine|boys|dogs|yell|hair|guys|kept|doer|fall|fell|head|shut|gift|hole|rest|numb|kick|lean|take|both|sick|fail|fade|took|miss|side|sigh|held|talk|last|plan|bore|hold|done|tall|teas|fact|boot|like|wife|rich|sign|book|wood|team|does|main|offs|tear|tore|torn|rode|dirt|gets|bone|joke|ride|make|told|play|died|tell|dies|tens|area|body|than|boat|line|guns|desk|that|what|kiss|them|they|gate|sang|then|plea|kill|face|sing|sung|eyes|thin|blue|deep|made|rung|ring|sirs|wide|he's|rang|moon|blow|eyed|sits|more|whys|dead|blew|days|this|left|grew|he's|size|rise|rose|whom|have|skin|most|late|grow|slam|road|game|tied|ties|arms|time|dark|rock|okay|ages|mens|roll|mans|tiny|slid|dads|airs|ok'd|tire|wets|ok'd|i'll|roof|slip|full|cuts|pool|slow|tone|bite|lips|cups|bits|room|olds|poor|bird|adds|ever|knew|hate|fuck|pops|even|tops|wipe|hits|once|west|hour|rows|rubs|toss|best|ones|only|from|runs|bend|bent|onto|open|move|town|free|pour|legs|rush|jump|snap|many|hill|less|maps|snow|keep|safe|much|soft|join|beer|i'll|beds|four|tree|same|sand|form|cops|must|year|cool|trip|lets|beat|mark|born|bear|with|come|save|know|true|sons|lock|song|soon|laws|came|outs|name|well|been|says|said|sort|feet|soul|high|yeah|were|hide|foot|turn|cold|wind|yard|twos|coat|food|over|hats|owns|ends|lady|aged|arts|else|long|flew|hurt|page|week|upon|lays|used|uses|hard|eggs|wins|very|mays|seas|pain|near|view|bars|weds|pull|edge|wrap|lies|bank|spin|ball|grey|seat|spun|lied|neck|push|wait|hope|bags|city|look|wake|spot|saws|woke|wear|pink|liar|eats|walk|need|sees|seen|puts|seem|wall|want|pair|gray|sell|will|flat|back|pale|sold|asks|wars|land|send|mean|warm|baby|sent|also|wash|away|here|easy|hung|sens|star|hers|aunt|palm|worn|life|meet|wore|east|live|news|five|wave|next|lost|lose|nice|ways|far|few|war|bad|bag|bar|wed|use|ups|art|was|two|try|are|bed|top|arm|wet|big|too|bit|tie|the|ten|tvs|tea|box|boy|sun|bus|but|buy|any|can|car|cat|and|son|cop|sos|cry|cup|cut|who|dad|sky|day|six|why|sit|sat|sir|die|did|dog|she|dry|sex|set|ear|ate|eat|see|saw|win|won|sea|egg|end|say|sad|ran|run|rub|row|eye|rid|ask|fed|fee|red|way|fit|fix|all|put|fly|for|pop|fun|get|got|god|pay|own|out|our|air|ors|one|old|ohs|gun|key|off|guy|now|not|nor|nod|nos|ago|new|hat|age|had|has|her|met|hey|may|hid|map|him|add|his|man|men|hit|mad|low|lot|hot|lip|how|lit|lie|kid|i'm|let|i'm|leg|i'd|i'd|ice|led|act|lay|law|ins|yes|yet|you|its|job|no|at|by|my|on|ha|do|ok|he|oh|is|tv|me|us|as|hi|go|if|of|am|up|to|we|so|in|or|it|be|an|i|a".split(
    "|"
)
```

Let's also create a checker that flags any "banned" words in a text:



```python
import re


def checker(text: str, vocab: list[str]) -> set[str]:
    """Check if a text contains any words that are not in the vocabulary."""
    text = re.sub(r"-", " ", text)
    words = re.sub(r"[^a-z' \n]+", "", text.lower()).split()
    return set(words) - set(vocab)
```

Let's test it out with the famous English language pangram:



```python
checker("The quick brown fox jumps over the lazy dog.", VOCAB)
```




    {'fox', 'lazy'}



You’ll see `{'fox', 'lazy'}` flagged. So far, so good.


## Attempt 1: Just Ask Nicely


Let's try with the obvious approach: directly asking GPT-4o to only use the 1,000 most common words in the English language.



```python
%load_ext dotenv
%dotenv
```


```python
from openai import OpenAI


client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input="Explain how airplanes work using only the 1,000 most common English words.",
)

output = response.output_text
print(output)
```

    Airplanes are machines that fly through the air. They have wings, engines, and a body where people or things can sit.
    
    The wings help lift the plane off the ground. As the plane moves fast on the ground, air goes over and under the wings. This makes the plane go up.
    
    The engines give the plane a strong push forward. This is called thrust. The engines use fuel to work and make the plane speed up.
    
    The body of the plane, called the fuselage, holds everything together. It has seats for people and space for bags or other things.
    
    The tail helps keep the plane steady and points it in the right way. It has small wings to help turn the plane.
    
    To steer the plane, pilots use controls to move parts on the wings and tail. These parts change how the air moves around the plane, helping it go up, down, or turn.
    
    Airplanes can fly because of the careful mix of lift from the wings, thrust from the engines, and control from the tail and pilots.



```python
checker(output, VOCAB)
```




    {'airplanes',
     'fuel',
     'fuselage',
     'mix',
     'pilots',
     'plane',
     'speed',
     'steady',
     'steer',
     'tail',
     'thrust'}



It tried. But our checker flagged words like "airplanes", "fuel", "fuselage", etc. Clearly, it didn't follow the rule. I suppose it was unreasonable to expect it just to know what the 1,000 most common words are.


**Result:** ❌


## Attempt 2: Penalize Uncommon Words with `logit_bias`


Let's try a different approach. OpenAI's API lets you "nudge" models away from certain words using `logit_bias`. We can try telling GPT-4o to strongly avoid every token not in our vocabulary.



```python
import tiktoken

from openai import OpenAI

encoding = tiktoken.encoding_for_model("gpt-4o")
token_ids = set(encoding.encode(" ".join(VOCAB) + " " + " ".join(VOCAB).title()))
logit_bias = {i: -100 for i in range(200_000) if i not in token_ids}

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain how airplanes work."}],
    logit_bias=logit_bias,
)

output = response.choices[0].message.content
print(output)
```

```console
Invalid 'logit_bias': too many properties. Expected an object with at most 1024 properties, but got an object with 194256 properties instead.
```


Okay, it seems like we ran into a hard limit: the API only allows biasing up to 1,024 tokens. Our banned list was way longer.


**Result:** ❌


## Attempt 3: Penalize Logits on Local Model


We can try the same method with a local Llama 3 model. No API limits here! We bias the model to strongly avoid all tokens not in the allowed list, although we explicitly allow variants like capitalizing the first letter for titles to allow the LLM to respond naturally. There are also some special tokens we want to permit that aren't in the vocabulary. These tokens include predefined symbols that act as markers, as well as punctuation.



```python
SPECIAL_TOKENS = [
    "<|start_header_id|>",
    "assistant",
    "<|end_header_id|>",
    "\n\n",
    ".",
    ",",
    " ",
    "<|eot_id|>",
]
```


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList


class LogitBiasProcessor(LogitsProcessor):
    """Use the logit_bias dictionary to bias the logits of the tokens."""

    def __init__(self, logit_bias: dict):
        self.token_ids = torch.tensor(list(logit_bias.keys()), dtype=torch.long)
        self.biases = torch.tensor(list(logit_bias.values()), dtype=torch.float)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.token_ids = self.token_ids.to(scores.device)
        self.biases = self.biases.to(scores.device)
        scores[:, self.token_ids] += self.biases
        return scores


device = "cuda"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)

inputs = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how an airplane works."},
    ],
    return_tensors="pt",
).to(device)
prompt_len = inputs.shape[1]

token_ids = set(
    tokenizer.encode(
        " ".join(VOCAB) + " " + " ".join(VOCAB).title() + " " + " ".join(SPECIAL_TOKENS)
    )
)
logit_bias = {i: -100 for i in range(len(tokenizer)) if i not in token_ids}

outputs = model.generate(
    inputs,
    max_new_tokens=100,
    logits_processor=LogitsProcessorList([LogitBiasProcessor(logit_bias)]),
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

output = (
    tokenizer.decode(output[:, prompt_len:][0])
    .rpartition("<|end_header_id|>")[2]
    .strip()
)
print(output)
```

    Sitting on a flight can be a great way to travel but have you ever wondered how an entire machine can lift off the ground and fly through the air with hundreds of people on board. Here's a simple and easy to understand break down of how an Air Plan works. It's a bit of a long story but I'll do my best to make it as simple as possible. An Air Plan is a machine that uses four forces of flight to lift off the ground and stay



```python
checker(output, VOCAB)
```




    {"here's", "it's"}



This kind of worked, but not perfectly. The model still slipped into banned contractions like _here's_ and _let's_ and broken-up words like _Air Plan_. That's because token-level filtering isn't quite enough. Banned contractions can sneak in as combinations of allowed tokens. And sometimes, the model starts a word with a valid token, but the only way to complete it is with a banned one, so it gets stuck and outputs the next best alternative within the limited vocabulary, which is often nonsensical.


**Result:** ❌


## Attempt 4: Prefix-Constrained Decoding using a Token Trie


Instead of filtering tokens, we can try controlling generation at the word level.

Here's how:

- We build a trie (prefix tree) of all allowed words.
- We use that to constrain generation using beam search.
- The model is only allowed to follow valid word paths in the trie.



```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)


class TokenTrieNode:
    """A node in the token trie."""

    def __init__(self):
        self.children = dict()
        self.is_end = False
        self.starts_with_space = False


class TokenTrie:
    """A token trie containing valid words."""

    def __init__(self):
        self.root = TokenTrieNode()
        self.max_depth = 0

    def insert(self, token_ids, raw_text):
        node = self.root

        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TokenTrieNode()
            node = node.children[token_id]
            node.starts_with_space = raw_text.startswith(" ")

        node.is_end = True
        self.max_depth = max(self.max_depth, len(token_ids))

    def get_next_tokens(self, prefix_ids, only_starts_with_space=False):
        node = self.root
        for token_id in prefix_ids:
            if token_id not in node.children:
                return []  # dead end
            node = node.children[token_id]
        if only_starts_with_space:
            return set(
                token_id
                for token_id, child in node.children.items()
                if child.starts_with_space
            )
        else:
            return set(node.children.keys())


inputs = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": "You are a helpful assistant that explains things concisely in one sentence using simple words.",
        },
        {
            "role": "user",
            "content": "Explain how an airplane works in one sentence.",
        },
    ],
    return_tensors="pt",
).to(device)
prompt_len = inputs.shape[1]

trie = TokenTrie()
for word in VOCAB + SPECIAL_TOKENS:
    variants = [word, word.title(), " " + word, " " + word.title()]

    for variant in variants:
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        trie.insert(token_ids, variant)


def prefix_allowed_tokens_fn(batch_id, input_ids):
    """A function that returns the allowed tokens for the next generation."""

    full_input = input_ids.tolist()
    generated = full_input[prompt_len:]
    allowed = set()

    # Try all possible prefixes up to max token length
    for i in range(trie.max_depth):
        prefix = generated[-i:]
        allowed.update(trie.get_next_tokens(prefix))

    # Always allow starting a new word from scratch
    if len(generated) < 6:  # let it generate header unconstrained
        allowed.update(trie.get_next_tokens([]))
    else:
        allowed.update(trie.get_next_tokens([], only_starts_with_space=True))
        for token in SPECIAL_TOKENS:
            allowed.add(tokenizer.encode(token, add_special_tokens=False)[0])

    return list(allowed) if allowed else [tokenizer.eos_token_id]


outputs = model.generate(
    inputs,
    max_new_tokens=150,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    num_beams=4,
    repetition_penalty=1.5,
)

output = (
    tokenizer.decode(outputs[0], skip_special_tokens=True)
    .rpartition("assistant")[2]
    .strip()
)
print(output)
```

    A powered flying machine, or more simply put, a flying machine, works by using the shape of its wings to create lift as it moves forward, allowing it to rise and stay in the air, while its engines or engines and control surfaces work together to control its flight path and direction.



```python
checker(output, VOCAB)
```




    set()



Look, no banned words! It seems that a combination of beam search and being greedy on the word level instead of the token level gives it enough flexibility to recover from mistakes.


**Result:** ✔️


## Additional Examples


Let's try this approach on a few more science-y topics. The results are a little long-winded but reasonably clear and completely in bounds.

Explain the International Space Station in one sentence.
: Able to hold up to six people at any given time, the space station is a large, state of the art, low Earth space station that serves as a home away from home in space, a place for space travelers to live and work for months at a time, while also serving as a center for Earth and space studies, as well as a proving ground for deep space travel.

Explain how a human cell works in one sentence.
: A human body is made up of tiny building blocks, called human or body calls, which have different parts that work together to keep us alive, like a tiny city with its own streets, houses, and services.

Explain how a nuclear reactor works in one sentence.
: In a power station, water is heated in a controlled way by the heat given off from tiny pieces of a special kind of metal that breaks down when it gets too hot, which then turns a wheel to make light and power for homes and businesses.
  
Explain how a transformer model works in one sentence.
: A key part of the work of a deep learning system known as a large language and other forms of machine learning as well, is that it works by taking in a piece of information, breaking it down into smaller parts, and then using these parts to create a new understanding of the information that can be used to answer a question, create a new piece of information, and so on, through the use of something called attention and a large number of layers that are trained on a large number of pieces of information in order to learn how to understand and work with the information in the way that a human would.

## References

[^thing-explainer]: Munroe, R. (2015). *Thing Explainer: Complicated Stuff in Simple Words.* Houghton Mifflin Harcourt.