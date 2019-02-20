import pandas as pd
import sys
import itertools


def load_gtf(gtf_path):
    '''
    Loads a GTF and returns a dataframe
    No filtering or anything performed.
    '''
    df = pd.read_csv(gtf_path, 
        sep='\t', 
        comment='#', 
        header=None,
        low_memory=False,
        names=['chr',
            'source',
            'feature',
            'start',
            'end',
            'score', 
            'strand', 
            'frame', 
            'attribute'])
    return df

def extract(attribute_field, fieldname):
    '''
    Parse the catch-call attribute field, and return the value of the field requested 
    with the `fieldname` variable.  If not found, return np.nan
    Used by the `extract_from_attributes` function
    '''
    contents = [x.strip() for x in attribute_field.split(';')]
    for c in contents:
        try:
            x,y = [s.strip() for s in c.split(' ')]
            if x == fieldname:
                return y[1:-1]
        except ValueError:
            return pd.np.nan
    return pd.np.nan


def extract_from_attributes(gtf, fieldname):
    '''
    Looks through the attribute data and finds the entry for 'fieldname'
    '''
    col = gtf['attribute'].apply(extract, args=(fieldname,))
    gtf[fieldname] = col
    return gtf


def get_exons(full_gtf, gene_name):
    '''
    Given a gene name, finds the first transcript and returns
    the constitutive exons of that transcript.  If no transcripts, return
    an empty dataframe
    '''
    # get all entries for this gene
    single_gene_gtf = gtf.loc[gtf['gene_name'] == gene_name]

    # get only the trascripts for this gene
    transcript_gtf = single_gene_gtf.loc[single_gene_gtf['feature'] == 'transcript']
    if transcript_gtf.shape[0] > 0:
        selected_transcript = transcript_gtf.iloc[0]['transcript_id'] # grab the first transcript
        # find all exons for this transcript:
        exons_gtf = gtf.loc[(gtf['feature'] == 'exon') & (gtf['transcript_id'] == selected_transcript)]
        return exons_gtf
    else:
        return pd.DataFrame()


def pick_overlapped_genes(gtf):
    '''
    Find the first pair of genes (technically transcripts) that overlap 
    and return a dataframe giving the exons of those transcripts.

    Does NOT actually check that the exons themselves overlap...so it's worth checking 
    that this is in fact the case.  Might add later, but not at the moment.
    '''
    # - gene on (+) strand that has *some* overlap with a gene on (-) strand (or vice versa)
    genes_of_interest = pd.DataFrame()

    found = False
    idx = pd.Index(range(gtf.shape[0]))
    i = 0
    strand_map = {'+': '-', '-':'+'}
    while (i < gtf.shape[0]) and not found:
        # go gene-by-gene.  Compare against the other genes
        other_rows = gtf.loc[gtf.index > i] # get the rows after
        row = gtf.iloc[i]
        start = row['start']
        end = row['end']
        strand = row['strand']
        f1 = other_rows['start'] < end
        f2 = other_rows['end'] > start

        # if there is an overlap:
        if pd.np.sum(f1 & f2) > 0:
            overlapping = other_rows.loc[f1 & f2]
            other_strand = strand_map[strand]
            other_strand_overlap = overlapping.loc[overlapping['strand'] == other_strand]
            if other_strand_overlap.shape[0] > 0:
                # there is an overlap in the gene coords.  We really need a transcript
                geneA_name = row['gene_name']
                geneB_name = other_strand_overlap.iloc[0]['gene_name'] # pick first.  plenty of genes that we don't need to cycle through possibly >1
                geneA_df = gtf.loc[gtf['gene_name'] == geneA_name] # get all features for this gene
                geneB_df = gtf.loc[gtf['gene_name'] == geneB_name] # get all features for other gene
                geneA_transcripts = geneA_df['transcript_id'].unique()
                geneB_transcripts = geneB_df['transcript_id'].unique()
                for t1_id, t2_id in itertools.product(geneA_transcripts, geneB_transcripts):
                    t1 = geneA_df.loc[(geneA_df['transcript_id'] == t1_id) & (geneA_df['feature'] == 'transcript')]
                    t2 = geneB_df.loc[(geneB_df['transcript_id'] == t2_id) & (geneB_df['feature'] == 'transcript')]
                    if (t1.shape[0] > 0) & (t2.shape[0] > 0):
                        t1 = t1.iloc[0]
                        t2 = t2.iloc[0]
                        if (t1.start < t2.end) & (t1.end > t2.start): # overlap in transcripts:
                            found = True
                            exonsA = geneA_df.loc[(geneA_df['transcript_id'] == t1_id) & (geneA_df['feature'] == 'exon')]
                            exonsB = geneB_df.loc[(geneB_df['transcript_id'] == t2_id) & (geneB_df['feature'] == 'exon')]
                            genes_of_interest = genes_of_interest.append(exonsA)
                            genes_of_interest = genes_of_interest.append(exonsB)
                            return genes_of_interest
        i += 1
    return genes_of_interest


def pick_non_overlapped_genes(gtf):
    '''
    Do a linear search and find:
     - gene on (+) strand that has no overlapping gene
     - gene on (-) strand that has no overlapping gene
     We don't actually care which genes we choose

     Return a dataframe that has the exons of those chosen transcripts
    '''

    # filter for gene features:
    gene_gtf = gtf.loc[gtf['feature'] == 'gene'].copy()
    gene_gtf = gene_gtf.reset_index()

    case1_found = False
    case2_found = False
    genes_of_interest = pd.DataFrame()

    idx = pd.Index(range(gene_gtf.shape[0]))
    i = 0
    strand_map = {'+': '-', '-':'+'}
    while (i < gene_gtf.shape[0]) and not all([case1_found, case2_found]):
        # go gene-by-gene.  Compare against the other genes
        other_rows = gene_gtf.loc[gene_gtf.index != i]
        row = gene_gtf.iloc[i]
        start = row['start']
        end = row['end']
        strand = row['strand']
        f1 = other_rows['start'] < end
        f2 = other_rows['end'] > start

        # if there is no overlap:
        if pd.np.sum(f1 & f2) == 0:
            gene_name = row['gene_name']
            if strand == '+' and not case1_found:
                exons_gtf = get_exons(gtf, gene_name)
                if exons_gtf.shape[0] > 0:
                    case1_found = True
                    genes_of_interest = genes_of_interest.append(exons_gtf)
            elif strand == '-' and not case2_found:
                exons_gtf = get_exons(gtf, gene_name)
                if exons_gtf.shape[0] > 0:
                    case2_found = True
                    genes_of_interest = genes_of_interest.append(exons_gtf)
            else:
                pass

        i += 1
    return genes_of_interest


if __name__ == '__main__':
    
    # cheap way of getting commandline params
    gtf_path = sys.argv[1]
    chrom = sys.argv[2]

    # load gtf and subset for the chosen chromosome
    gtf = load_gtf(gtf_path)
    gtf = gtf.loc[gtf['chr'] == chrom]

    # further subset for only 'gene' features
    #genes_gtf = gtf.loc[gtf['feature'] == 'gene'].copy()

    # add extra fields derived from attribute col:
    gtf = extract_from_attributes(gtf, 'gene_name')
    gtf = extract_from_attributes(gtf, 'gene_id')
    gtf = extract_from_attributes(gtf, 'transcript_id')
    gtf = gtf.reset_index() # change index to be 0,1,...,N

    non_overlapped = pick_non_overlapped_genes(gtf)
    overlapped = pick_overlapped_genes(gtf)
    genes_of_interest = pd.concat([non_overlapped, overlapped])
    genes_of_interest = extract_from_attributes(genes_of_interest, 'gene_name')
    genes_of_interest = extract_from_attributes(genes_of_interest, 'gene_id')
    genes_of_interest = extract_from_attributes(genes_of_interest, 'transcript_id')
    genes_of_interest = extract_from_attributes(genes_of_interest, 'exon_number')
    selected_cols = ['chr', 'feature', 'start','end', 'strand', 'gene_name', 'exon_number', 'gene_id', 'transcript_id']
    genes_of_interest[selected_cols].to_csv('selected_genes.chr%s.tsv' % chrom, sep='\t', index=False)

    # make an alternate version ready for bedtools:
    genes_of_interest['seq_name'] = genes_of_interest['transcript_id'] + '_' + genes_of_interest['exon_number']
    genes_of_interest['score'] = 1
    selected_cols = ['chr', 'start','end', 'seq_name','score','strand']
    bed_ready = genes_of_interest[selected_cols]
    bed_ready['start'] = bed_ready['start'] - 1 # BED is a 0-based system, GTF was 1-based
    bed_ready['end'] = bed_ready['end'] - 1 
    bed_ready.to_csv('bedready.chr%s.tsv' % chrom, sep='\t', index=False, header=False)







